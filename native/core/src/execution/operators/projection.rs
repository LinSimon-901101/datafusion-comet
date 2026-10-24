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

use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_err, Result};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::expressions::Column;
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
use datafusion_comet_spark_expr::ScalarSubqueryConsumer;
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
        let input = bound_projection_input(
            Arc::clone(&child.native_plan),
            &exprs,
            planner.session_ctx().copied_config().batch_size(),
        );
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
    use arrow::array::{Array, Int64Array, StringArray, StructArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::common::ScalarValue;
    use datafusion::datasource::memory::MemorySourceConfig;
    use datafusion::execution::context::SessionContext;
    use datafusion::physical_plan::{ChildrenPropertiesMode, ReplaceChildrenOptions};
    use datafusion_comet_spark_expr::{GetStructField, Subquery};
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
        let projection = ProjectionExec::try_new(exprs.iter().cloned(), Arc::clone(&bounded)).unwrap();
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
