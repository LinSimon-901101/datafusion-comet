// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Projection operator builder

use std::sync::Arc;

use datafusion::common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion::common::{internal_err, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::{Column, Literal};
use datafusion::physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_plan::execution_plan::CardinalityEffect;
use datafusion::physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion::physical_plan::metrics::{ExecutionPlanMetricsSet, MetricsSet, SplitMetrics};
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::stream::BatchSplitStream;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, PlanProperties, SendableRecordBatchStream,
};
use datafusion_comet_proto::spark_operator::Operator;
use datafusion_comet_spark_expr::{GetStructField, ScalarSubqueryConsumer};
use jni::objects::{Global, JObject};

use crate::{
    execution::{
        planner::{operator_registry::OperatorBuilder, PhysicalPlanner, PlanCreationResult},
        spark_plan::SparkPlan,
    },
    extract_op,
};

/// Builder for Projection operators
pub struct ProjectionBuilder;

impl OperatorBuilder for ProjectionBuilder {
    fn build(
        &self,
        spark_plan: &Operator,
        inputs: &mut Vec<Arc<Global<JObject<'static>>>>,
        partition_count: usize,
        planner: &PhysicalPlanner,
    ) -> PlanCreationResult {
        let project = extract_op!(spark_plan, Projection);
        let children = &spark_plan.children;

        assert_eq!(children.len(), 1);
        let (scans, shuffle_scans, mut child) =
            planner.create_plan(&children[0], inputs, partition_count)?;

        // Create projection expressions
        let exprs: Result<Vec<_>, _> = project
            .project_list
            .iter()
            .enumerate()
            .map(|(idx, expr)| {
                planner
                    .create_expr(expr, child.schema())
                    .map(|r| ProjectionExpr::new(r, format!("col_{idx}")))
            })
            .collect();

        let mut exprs = ProjectionExprs::from(exprs?);
        if let Some(filter) = child.native_plan.downcast_ref::<FilterExec>() {
            if exprs.iter().all(|expr| expr.expr.is::<Column>()) {
                let indices = exprs.column_indices();
                if indices.len() < child.schema().fields().len() {
                    // Filter each required output column once, then restore its order and aliases.
                    // Keep both native plans so Spark metrics describe the executed work.
                    let mapping = ProjectionExprs::from_indices(&indices, &child.schema());
                    exprs = exprs.try_map_exprs(|expr| mapping.project_expr(&expr))?;
                    let filter = FilterExecBuilder::from(filter)
                        .apply_projection(Some(indices))?
                        .build()?;
                    Arc::make_mut(&mut child).native_plan = Arc::new(filter);
                }
            }
        }
        let batch_size = planner.session_ctx().copied_config().batch_size();
        let (input, exprs) =
            inline_scalar_alias_projection(Arc::clone(&child.native_plan), exprs, batch_size)?;
        let input = bound_projection_input(input, &exprs, batch_size);
        let projection = Arc::new(ProjectionExec::try_new(exprs.iter().cloned(), input)?);

        let spark_plan = if child.plan_id == spark_plan.plan_id {
            let mut additional_native_plans = vec![Arc::clone(&child.native_plan)];
            additional_native_plans.extend(child.additional_native_plans.iter().cloned());
            Arc::new(SparkPlan::new_with_additional(
                spark_plan.plan_id,
                projection,
                child.children.clone(),
                additional_native_plans,
            ))
        } else {
            Arc::new(SparkPlan::new(spark_plan.plan_id, projection, vec![child]))
        };

        Ok((scans, shuffle_scans, spark_plan))
    }
}

/// Consume a retained literal-field alias before broadcasting it. Only remove an immediate
/// capped, read-only producer; computed projections and other operator boundaries stay intact.
fn inline_scalar_alias_projection(
    mut input: Arc<dyn ExecutionPlan>,
    mut exprs: ProjectionExprs,
    batch_size: usize,
) -> Result<(Arc<dyn ExecutionPlan>, ProjectionExprs)> {
    while let Some(inner) = input.downcast_ref::<ProjectionExec>() {
        let Some(cap) = inner.input().downcast_ref::<ProjectionInputBatchExec>() else {
            break;
        };
        if !inner.expr().iter().all(|expr| {
            expr.expr.is::<Column>()
                || expr.expr.is::<Literal>()
                || expr
                    .expr
                    .downcast_ref::<GetStructField>()
                    .is_some_and(|field| field.literal_field().is_some())
        }) {
            break;
        }
        // Do not erase externally supplied field/schema metadata.
        let derived = inner
            .projection_expr()
            .make_projector(&inner.input().schema())?;
        if derived.output_schema() != &inner.schema() {
            break;
        }
        let unwrapped = exprs.clone().try_map_exprs(|expr| {
            Ok(expr
                .transform_up(|expr| {
                    Ok(match expr.downcast_ref::<ScalarSubqueryConsumer>() {
                        Some(consumer) => Transformed::yes(Arc::clone(consumer.inner())),
                        None => Transformed::no(expr),
                    })
                })?
                .data)
        })?;
        let mapped = inner.projection_expr().try_merge(&unwrapped)?;
        let schema = cap.input.schema();
        if !mapped.iter().all(|expr| {
            ScalarSubqueryConsumer::supports_sliced_evaluation(&expr.expr, &schema, batch_size)
        }) {
            break;
        }
        exprs = mapped.try_map_exprs(|expr| {
            Ok(expr
                .transform_up(|expr| {
                    Ok(Transformed::yes(ScalarSubqueryConsumer::wrap_with_schema(
                        expr,
                        Arc::clone(&schema),
                    )))
                })?
                .data)
        })?;
        input = Arc::clone(&cap.input);
    }
    Ok((input, exprs))
}

fn bound_projection_input(
    input: Arc<dyn ExecutionPlan>,
    exprs: &ProjectionExprs,
    batch_size: usize,
) -> Arc<dyn ExecutionPlan> {
    let max_rows = exprs
        .iter()
        .filter_map(|expr| {
            ScalarSubqueryConsumer::projection_max_rows(&expr.expr, input.schema().as_ref())
        })
        .min();
    match max_rows {
        Some(max_rows) if max_rows < batch_size => {
            Arc::new(ProjectionInputBatchExec::new(input, max_rows))
        }
        _ => input,
    }
}

/// Split before evaluating the projection, so aliases never materialize an oversized array.
#[derive(Debug)]
struct ProjectionInputBatchExec {
    input: Arc<dyn ExecutionPlan>,
    max_rows: usize,
    metrics: ExecutionPlanMetricsSet,
}

impl ProjectionInputBatchExec {
    fn new(input: Arc<dyn ExecutionPlan>, max_rows: usize) -> Self {
        Self {
            input,
            max_rows: max_rows.max(1),
            metrics: ExecutionPlanMetricsSet::new(),
        }
    }
}

impl DisplayAs for ProjectionInputBatchExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "ProjectionInputBatchExec: max_rows={}", self.max_rows)
    }
}

impl ExecutionPlan for ProjectionInputBatchExec {
    fn name(&self) -> &str {
        "ProjectionInputBatchExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        self.input.properties()
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }
    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::Equal
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("ProjectionInputBatchExec requires one child");
        }
        Ok(Arc::new(Self::new(children.remove(0), self.max_rows)))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        Ok(Box::pin(BatchSplitStream::new(
            self.input.execute(partition, context)?,
            self.max_rows,
            SplitMetrics::new(&self.metrics, partition),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Array, BooleanArray, Int64Array, StringArray, StructArray};
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::{config::ConfigOptions, ScalarValue};
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::context::SessionContext;
    use datafusion::logical_expr::{Operator, ScalarUDF};
    use datafusion::physical_expr::expressions::{BinaryExpr, IsNotNullExpr, IsNullExpr, Literal};
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
    use datafusion_comet_spark_expr::{
        Cast, EvalMode, GetStructField, IfExpr, RandExpr, SparkCastOptions, Subquery,
    };
    use futures::TryStreamExt;

    fn source(rows: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..rows as i64))],
        )
        .unwrap();
        MemorySourceConfig::try_new_exec(&[vec![batch]], schema, None).unwrap()
    }

    #[tokio::test]
    async fn scalar_output_projection_splits_before_materializing_aliases() {
        let input = source(128);
        let scalar = ScalarValue::Struct(Arc::new(StructArray::new(
            vec![Field::new("s", DataType::Utf8, true)].into(),
            vec![Arc::new(StringArray::from(vec!["x".repeat(65536)]))],
            None,
        )));
        let field: Arc<dyn PhysicalExpr> = Arc::new(GetStructField::new(
            Arc::new(Subquery::resolved_struct_literal(scalar)),
            0,
        ));
        let exprs = ProjectionExprs::from(vec![
            ProjectionExpr::new(field, "alias".to_owned()),
            ProjectionExpr::new(Arc::new(Column::new("id", 0)), "id".to_owned()),
        ]);
        let bounded = bound_projection_input(Arc::clone(&input), &exprs, 8192);
        assert!(bounded.is::<ProjectionInputBatchExec>());
        assert!(Arc::ptr_eq(bounded.properties(), input.properties()));
        assert_eq!(bounded.maintains_input_order(), vec![true]);
        let projection =
            ProjectionExec::try_new(exprs.iter().cloned(), Arc::clone(&bounded)).unwrap();
        let batches = projection
            .execute(0, SessionContext::new().task_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert!(batches.len() > 1);
        let num_batches = batches.len();
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            128
        );
        let mut ids: Vec<i64> = Vec::new();
        for batch in batches {
            assert!(batch.num_rows() <= 16);
            let values = batch
                .column(0)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap();
            assert_eq!(values.len(), batch.num_rows());
            assert!(values
                .iter()
                .all(|value| value.is_some_and(|value| value.len() == 65536)));
            ids.extend(
                batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values(),
            );
        }
        assert_eq!(ids, (0..128i64).collect::<Vec<_>>());
        assert_eq!(
            bounded
                .metrics()
                .unwrap()
                .sum_by_name("batches_split")
                .unwrap()
                .as_usize(),
            num_batches
        );
        let rewritten = bounded
            .replace_children(
                vec![source(0)],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
            .unwrap();
        assert!(rewritten.is::<ProjectionInputBatchExec>());
        let empty = rewritten
            .execute(0, SessionContext::new().task_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(empty.len(), 1);
        assert_eq!(empty.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
    }

    #[tokio::test]
    async fn bounded_conditional_output_keeps_one_full_projection_batch() {
        let rows = 8192;
        let input = source(rows);
        let schema = input.schema();
        let scalar = ScalarValue::Struct(Arc::new(StructArray::new(
            vec![Field::new("n", DataType::Int64, true)].into(),
            vec![Arc::new(Int64Array::from(vec![262144]))],
            None,
        )));
        let field: Arc<dyn PhysicalExpr> = Arc::new(GetStructField::new(
            Arc::new(Subquery::resolved_struct_literal(scalar)),
            0,
        ));
        let formatted = ScalarSubqueryConsumer::wrap_with_schema(
            Arc::new(Cast::new(
                ScalarSubqueryConsumer::wrap_with_schema(field, Arc::clone(&schema)),
                DataType::Utf8,
                SparkCastOptions::new(EvalMode::Legacy, "UTC"),
                None,
                None,
            )),
            Arc::clone(&schema),
        );
        let condition = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("id", 0)),
            Operator::Lt,
            Arc::new(Literal::new(ScalarValue::Int64(Some((rows / 2) as i64)))),
        ));
        let conditional = ScalarSubqueryConsumer::wrap_with_schema(
            Arc::new(IfExpr::new(
                condition,
                formatted,
                Arc::new(Literal::new(ScalarValue::Utf8(Some("x".to_owned())))),
            )),
            schema,
        );
        let exprs = ProjectionExprs::from(vec![
            ProjectionExpr::new(conditional, "value".to_owned()),
            ProjectionExpr::new(Arc::new(Column::new("id", 0)), "id".to_owned()),
        ]);
        // Exercise the same input cap and ProjectionExec construction as ProjectionBuilder.
        let bounded = bound_projection_input(input, &exprs, rows);
        let projection = ProjectionExec::try_new(exprs.iter().cloned(), bounded).unwrap();
        let batches = projection
            .execute(0, SessionContext::new().task_ctx())
            .unwrap()
            .try_collect::<Vec<_>>()
            .await
            .unwrap();
        assert_eq!(
            batches.len(),
            1,
            "a bounded choice must not become one-row batches"
        );
        let batch = &batches[0];
        assert_eq!(batch.num_rows(), rows);
        let values = batch
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        let ids = batch
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        for row in 0..rows {
            assert_eq!(ids.value(row), row as i64);
            assert_eq!(
                values.value(row),
                if row < rows / 2 { "262144" } else { "x" }
            );
        }
    }

    #[tokio::test]
    async fn retained_nested_scalar_alias_is_consumed_before_broadcast() {
        let rows = 8192;
        for whole_struct in [false, true] {
            for parent_valid in [false, true] {
                for nested_valid in [false, true] {
                    for leaf_valid in [false, true] {
                        let input = source(rows);
                        let fields = vec![
                            Field::new("text", DataType::Utf8, true),
                            Field::new("number", DataType::Int64, false).with_metadata(
                                [("PARQUET:field_id".to_owned(), "43".to_owned())].into(),
                            ),
                        ]
                        .into();
                        let nested = Arc::new(StructArray::new(
                            fields,
                            vec![
                                Arc::new(StringArray::from(vec![
                                    leaf_valid.then(|| "x".repeat(65536))
                                ])),
                                Arc::new(Int64Array::from(vec![17])),
                            ],
                            Some(NullBuffer::from(vec![nested_valid])),
                        ));
                        let root: Arc<dyn PhysicalExpr> =
                            Arc::new(Subquery::resolved_struct_literal(ScalarValue::Struct(
                                Arc::new(StructArray::new(
                                    vec![Field::new("nested", nested.data_type().clone(), true)]
                                        .into(),
                                    vec![nested],
                                    Some(NullBuffer::from(vec![parent_valid])),
                                )),
                            )));
                        let alias = if whole_struct {
                            root
                        } else {
                            Arc::new(GetStructField::new(root, 0)) as Arc<dyn PhysicalExpr>
                        };
                        // Keep the producer genuinely capped even when the tested alias is NULL.
                        let ballast: Arc<dyn PhysicalExpr> =
                            Arc::new(Subquery::resolved_struct_literal(ScalarValue::Struct(
                                Arc::new(StructArray::new(
                                    vec![Field::new("unused", DataType::Utf8, false)].into(),
                                    vec![Arc::new(StringArray::from(vec!["b".repeat(65536)]))],
                                    None,
                                )),
                            )));
                        let producer_exprs = ProjectionExprs::from(vec![
                            ProjectionExpr::new(alias, "alias"),
                            ProjectionExpr::new(Arc::new(Column::new("id", 0)), "id"),
                            ProjectionExpr::new(ballast, "unused"),
                        ]);
                        let capped =
                            bound_projection_input(Arc::clone(&input), &producer_exprs, rows);
                        assert!(capped.is::<ProjectionInputBatchExec>());
                        let producer: Arc<dyn ExecutionPlan> = Arc::new(
                            ProjectionExec::try_new(producer_exprs.iter().cloned(), capped)
                                .unwrap(),
                        );
                        let alias: Arc<dyn PhysicalExpr> = Arc::new(Column::new("alias", 0));
                        let nested = if whole_struct {
                            Arc::new(GetStructField::new(alias, 0)) as Arc<dyn PhysicalExpr>
                        } else {
                            alias
                        };
                        let text: Arc<dyn PhysicalExpr> =
                            Arc::new(GetStructField::new(Arc::clone(&nested), 0));
                        let outer = ProjectionExprs::from(vec![
                            ProjectionExpr::new(
                                Arc::new(IsNotNullExpr::new(Arc::clone(&text))),
                                "present",
                            ),
                            ProjectionExpr::new(Arc::new(IsNullExpr::new(text)), "missing"),
                            ProjectionExpr::new(Arc::new(GetStructField::new(nested, 1)), "number"),
                            ProjectionExpr::new(Arc::new(Column::new("id", 1)), "id"),
                        ]);
                        let expected_schema =
                            ProjectionExec::try_new(outer.iter().cloned(), Arc::clone(&producer))
                                .unwrap()
                                .schema();
                        let (base, mapped) =
                            inline_scalar_alias_projection(producer, outer, rows).unwrap();
                        assert!(
                            Arc::ptr_eq(&base, &input),
                            "retire the producer and its obsolete cap"
                        );
                        assert!(mapped.as_ref()[0].expr.is::<ScalarSubqueryConsumer>());
                        assert!(mapped.as_ref()[1].expr.is::<ScalarSubqueryConsumer>());
                        let bounded = bound_projection_input(base, &mapped, rows);
                        assert!(
                            Arc::ptr_eq(&bounded, &input),
                            "narrow results keep the full input batch"
                        );
                        let projection =
                            ProjectionExec::try_new(mapped.iter().cloned(), bounded).unwrap();
                        assert_eq!(projection.schema(), expected_schema);
                        let parents = mapped.as_ref()[2].expr.children();
                        let DataType::Struct(fields) =
                            parents[0].data_type(&input.schema()).unwrap()
                        else {
                            panic!("the selected numeric leaf keeps its nested struct type");
                        };
                        assert_eq!(fields[1].metadata()["PARQUET:field_id"], "43");
                        let batches = projection
                            .execute(0, SessionContext::new().task_ctx())
                            .unwrap()
                            .try_collect::<Vec<_>>()
                            .await
                            .unwrap();
                        assert_eq!(batches.len(), 1);
                        assert_eq!(batches[0].num_rows(), rows);
                        let present = batches[0]
                            .column(0)
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap();
                        let missing = batches[0]
                            .column(1)
                            .as_any()
                            .downcast_ref::<BooleanArray>()
                            .unwrap();
                        let numbers = batches[0]
                            .column(2)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        let ids = batches[0]
                            .column(3)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        for row in 0..rows {
                            assert_eq!(
                                present.value(row),
                                parent_valid && nested_valid && leaf_valid
                            );
                            assert_eq!(
                                missing.value(row),
                                !(parent_valid && nested_valid && leaf_valid)
                            );
                            assert_eq!(numbers.is_valid(row), parent_valid && nested_valid);
                            if numbers.is_valid(row) {
                                assert_eq!(numbers.value(row), 17);
                            }
                            assert_eq!(ids.value(row), row as i64);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn alias_inlining_keeps_computed_opaque_and_metadata_boundaries() {
        let input = source(32);
        let field: Arc<dyn PhysicalExpr> = Arc::new(GetStructField::new(
            Arc::new(Subquery::resolved_struct_literal(ScalarValue::Struct(
                Arc::new(StructArray::new(
                    vec![Field::new("s", DataType::Utf8, false)].into(),
                    vec![Arc::new(StringArray::from(vec!["x".repeat(65536)]))],
                    None,
                )),
            ))),
            0,
        ));
        let pure = ProjectionExprs::from(vec![
            ProjectionExpr::new(Arc::clone(&field), "alias"),
            ProjectionExpr::new(Arc::new(Column::new("id", 0)), "id"),
        ]);
        let cap = bound_projection_input(Arc::clone(&input), &pure, 8192);
        assert!(cap.is::<ProjectionInputBatchExec>());
        let producer: Arc<dyn ExecutionPlan> =
            Arc::new(ProjectionExec::try_new(pure.iter().cloned(), Arc::clone(&cap)).unwrap());
        let reducer = || {
            ProjectionExprs::from(vec![ProjectionExpr::new(
                Arc::new(IsNotNullExpr::new(Arc::new(Column::new("alias", 0)))),
                "present",
            )])
        };

        // An opaque/stateful outer tree must retain the original producer and batch boundary.
        let opaque = ProjectionExprs::from(vec![
            reducer().as_ref()[0].clone(),
            ProjectionExpr::new(Arc::new(RandExpr::new(42)), "random"),
        ]);
        let (kept, _) =
            inline_scalar_alias_projection(Arc::clone(&producer), opaque, 8192).unwrap();
        assert!(Arc::ptr_eq(&kept, &producer));

        // Retiring the cap must not expose independent unmarked payloads or growing siblings.
        let growing: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "repeat",
            Arc::new(ScalarUDF::new_from_impl(
                datafusion::functions::string::repeat::RepeatFunc::new(),
            )),
            vec![
                Arc::new(Literal::new(ScalarValue::Utf8(Some("x".to_owned())))),
                Arc::new(Column::new("id", 1)),
            ],
            Arc::new(Field::new("repeat", DataType::Utf8, true)),
            Arc::new(ConfigOptions::default()),
        ));
        let wide = ScalarValue::Utf8(Some("w".repeat(65536)));
        let ordinary_field: Arc<dyn PhysicalExpr> = Arc::new(GetStructField::new(
            Arc::new(Literal::new(ScalarValue::Struct(Arc::new(
                StructArray::new(
                    vec![Field::new("ordinary", DataType::Utf8, false)].into(),
                    vec![wide.to_array().unwrap()],
                    None,
                ),
            )))),
            0,
        ));
        let small: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion::physical_expr::expressions::CastExpr::new(
                Arc::new(Column::new("id", 1)),
                DataType::Utf8,
                None,
            ));
        for (value, inline) in [
            (growing, false),
            (
                Arc::new(Literal::new(wide.clone())) as Arc<dyn PhysicalExpr>,
                false,
            ),
            (ordinary_field, false),
            (
                Arc::new(Literal::new(ScalarValue::List(ScalarValue::new_list(
                    &[wide],
                    &DataType::Utf8,
                    false,
                )))) as Arc<dyn PhysicalExpr>,
                false,
            ),
            (
                Arc::new(Subquery::new(0, 0, DataType::Utf8)) as Arc<dyn PhysicalExpr>,
                false,
            ),
            (
                Arc::new(Subquery::new(0, 0, DataType::Binary)) as Arc<dyn PhysicalExpr>,
                false,
            ),
            (
                Arc::new(Subquery::new(0, 0, DataType::Int64)) as Arc<dyn PhysicalExpr>,
                true,
            ),
            (
                Arc::new(Literal::new(ScalarValue::List(ScalarValue::new_list(
                    &[
                        ScalarValue::Int64(Some(1)),
                        ScalarValue::Int64(Some(2)),
                        ScalarValue::Int64(Some(3)),
                    ],
                    &DataType::Int64,
                    false,
                )))) as Arc<dyn PhysicalExpr>,
                true,
            ),
            (small, true),
        ] {
            let outer = ProjectionExprs::from(vec![
                reducer().as_ref()[0].clone(),
                ProjectionExpr::new(value, "sibling"),
            ]);
            let (base, _) =
                inline_scalar_alias_projection(Arc::clone(&producer), outer, 8192).unwrap();
            assert_eq!(Arc::ptr_eq(&base, &input), inline);
            assert_eq!(Arc::ptr_eq(&base, &producer), !inline);
        }
        let output = ProjectionExprs::from(vec![ProjectionExpr::new(
            Arc::new(Column::new("alias", 0)),
            "wide",
        )]);
        let (base, mapped) =
            inline_scalar_alias_projection(Arc::clone(&producer), output, 8192).unwrap();
        assert!(Arc::ptr_eq(&base, &input));
        assert!(bound_projection_input(base, &mapped, 8192).is::<ProjectionInputBatchExec>());

        // Even a pure computed producer may throw or change work when duplicated or removed.
        let computed = ProjectionExprs::from(vec![ProjectionExpr::new(
            Arc::new(IfExpr::new(
                Arc::new(BinaryExpr::new(
                    Arc::new(Column::new("id", 0)),
                    Operator::Lt,
                    Arc::new(Literal::new(ScalarValue::Int64(Some(16)))),
                )),
                field,
                Arc::new(Literal::new(ScalarValue::Utf8(Some("small".to_owned())))),
            )),
            "alias",
        )]);
        let computed_input = bound_projection_input(Arc::clone(&input), &computed, 8192);
        assert!(computed_input.is::<ProjectionInputBatchExec>());
        let computed: Arc<dyn ExecutionPlan> =
            Arc::new(ProjectionExec::try_new(computed.iter().cloned(), computed_input).unwrap());
        let (kept, _) =
            inline_scalar_alias_projection(Arc::clone(&computed), reducer(), 8192).unwrap();
        assert!(Arc::ptr_eq(&kept, &computed));

        // Public metadata overrides cannot be reconstructed from the producer expressions.
        let overridden_schema = Schema::new_with_metadata(
            vec![
                Field::new("alias", DataType::Utf8, false)
                    .with_metadata([("comment".to_owned(), "override".to_owned())].into()),
                Field::new("id", DataType::Int64, false),
            ],
            [("schema-marker".to_owned(), "keep".to_owned())].into(),
        );
        let overridden: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new_with_schema_metadata(
                pure.iter().cloned(),
                cap,
                &overridden_schema,
            )
            .unwrap(),
        );
        let (kept, _) =
            inline_scalar_alias_projection(Arc::clone(&overridden), reducer(), 8192).unwrap();
        assert!(Arc::ptr_eq(&kept, &overridden));
        assert_eq!(kept.schema().field(0).metadata()["comment"], "override");
        assert_eq!(kept.schema().metadata()["schema-marker"], "keep");
    }

    #[test]
    fn small_scalar_and_ordinary_null_outputs_keep_original_input() {
        let input = source(128);
        let dt = DataType::Struct(vec![Field::new("s", DataType::Utf8, true)].into());
        let small: Arc<dyn PhysicalExpr> = Arc::new(Subquery::resolved_struct_literal(
            ScalarValue::Struct(Arc::new(StructArray::new(
                vec![Field::new("s", DataType::Utf8, true)].into(),
                vec![Arc::new(StringArray::from(vec!["kept"]))],
                None,
            ))),
        ));
        let ordinary: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion::physical_expr::expressions::Literal::new(
                ScalarValue::try_from(&dt).unwrap(),
            ));
        for expr in [small, ordinary] {
            let exprs = ProjectionExprs::from(vec![ProjectionExpr::new(expr, "value".to_owned())]);
            let bounded = bound_projection_input(Arc::clone(&input), &exprs, 8192);
            assert!(Arc::ptr_eq(&bounded, &input));
        }
    }
}
