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

use crate::array_funcs::NestedPredicate;
use crate::{
    ArrayInsert, AtLeastNNonNulls, CaseWhenExpr, Cast, CheckOverflow, CheckedBinaryExpr,
    CreateNamedStruct, DecimalRescaleCheckOverflow, FromJson, GetArrayStructFields, GetStructField,
    IfExpr, ListExtract, NegativeExpr, NormalizeNaNAndZero, NormalizeNestedFloats,
    NullShortCircuit, RLike, SparkArraysZipFunc, SparkComparison, Subquery, ToCsv, ToJson,
};
use arrow::array::{new_empty_array, Array, ArrayData, ArrayRef};
use arrow::compute::concat;
use arrow::datatypes::{DataType, FieldRef, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::{exec_err, internal_err, Result, ScalarValue};
use datafusion::logical_expr::{ColumnarValue, Volatility};
use datafusion::physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
};
use datafusion::physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Evaluates scalar-struct consumers before repeating large variable-width fields.
///
/// GetStructField retains its batch-sized array contract. Constant consumers run on one row;
/// pure row-dependent consumers run on bounded slices before their results are concatenated.
/// Nothing is evaluated during planning, preserving conditional and ANSI NULL short-circuiting.
#[derive(Eq)]
pub struct ScalarSubqueryConsumer {
    inner: Arc<dyn PhysicalExpr>,
    constant: bool,
    max_rows: usize,
    input_schema: SchemaRef,
}

// Limit repeated scalar input payload, not the size of the user's final output. A single scalar
// can exceed this budget; in that case consume one row at a time.
const SCALAR_INPUT_BYTES: usize = 1024 * 1024;

impl ScalarSubqueryConsumer {
    pub fn inner(&self) -> &Arc<dyn PhysicalExpr> {
        &self.inner
    }

    /// Keep direct struct output metric labels bounded. Apply only at a projection root,
    /// after field selection and input sizing have inspected the original literal.
    pub fn wrap_projection_literal(
        inner: Arc<dyn PhysicalExpr>,
        schema: SchemaRef,
    ) -> Arc<dyn PhysicalExpr> {
        if inner
            .downcast_ref::<Literal>()
            .is_some_and(Subquery::is_resolved_struct_literal)
        {
            Arc::new(Self {
                inner,
                constant: true,
                max_rows: 1,
                input_schema: schema,
            })
        } else {
            inner
        }
    }

    /// Let a later projection inline a retained whole-struct alias before broadcasting it.
    pub fn projection_literal(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.inner
            .downcast_ref::<Literal>()
            .filter(|literal| Subquery::is_resolved_struct_literal(literal))
            .map(|_| Arc::clone(&self.inner))
    }

    /// Inlining retires an input cap. Marked roots regain their consumer guard; unmarked
    /// siblings must be pure and fit a full batch without relying on the retired cap.
    pub fn supports_sliced_evaluation(
        expr: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
        batch_size: usize,
    ) -> bool {
        consumer_properties(expr, schema).is_some_and(|properties| {
            properties.has_scalar_struct
                || (!properties.unbounded_output
                    && properties.bytes <= SCALAR_INPUT_BYTES / batch_size.max(1))
        })
    }

    /// Cap a producing projection's input before a scalar value becomes an output column.
    /// Reducing expressions and ordinary literals/columns retain their normal batch size.
    pub fn projection_max_rows(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<usize> {
        let properties = consumer_properties(expr, schema)?;
        if !properties.has_scalar_struct
            || fixed_width_bytes(&expr.data_type(schema).ok()?).is_some()
        {
            return None;
        }
        Some(
            bounded_output_bytes(expr, schema)
                .map(|bytes| (SCALAR_INPUT_BYTES / bytes.max(1)).max(1))
                .unwrap_or(1),
        )
    }

    pub fn wrap(inner: Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr> {
        Self::wrap_with_schema(inner, Arc::new(Schema::empty()))
    }

    pub fn wrap_with_schema(
        inner: Arc<dyn PhysicalExpr>,
        input_schema: SchemaRef,
    ) -> Arc<dyn PhysicalExpr> {
        // Keep literals and direct field chains visible so a narrow field can be selected
        // before broadcasting. A field read over a computed struct is itself a consumer.
        if inner.is::<Literal>()
            || inner.is::<Self>()
            || inner
                .downcast_ref::<GetStructField>()
                .is_some_and(|field| field.literal_field().is_some())
        {
            return inner;
        }
        if let Some(properties) = consumer_properties(&inner, &input_schema) {
            if properties.has_scalar_struct
                && (properties.constant || properties.bytes != 0 || properties.unbounded_output)
            {
                return Arc::new(Self {
                    inner,
                    input_schema,
                    constant: properties.constant,
                    max_rows: if properties.unbounded_output {
                        1
                    } else {
                        (SCALAR_INPUT_BYTES / properties.bytes.max(1)).max(1)
                    },
                });
            }
        }
        inner
    }
}

struct ConsumerProperties {
    constant: bool,
    has_scalar_struct: bool,
    bytes: usize,
    unbounded_output: bool,
}

/// Explicitly recognize pure row-local trees. Unknown nodes and volatile functions cannot safely
/// be evaluated on a different batch. Columns require sliced evaluation.
fn consumer_properties(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Option<ConsumerProperties> {
    if let Some((field, value)) = expr
        .downcast_ref::<GetStructField>()
        .and_then(|field| field.literal_field().map(|value| (field, value)))
    {
        return Some(ConsumerProperties {
            constant: true,
            has_scalar_struct: field.is_subquery_field(),
            bytes: variable_bytes(&value.to_data()),
            unbounded_output: false,
        });
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        let bytes = if fixed_width_bytes(&literal.value().data_type()).is_some() {
            0
        } else {
            literal_bytes(literal.value())?
        };
        return Some(ConsumerProperties {
            constant: true,
            has_scalar_struct: Subquery::is_resolved_struct_literal(literal),
            bytes,
            unbounded_output: false,
        });
    }
    // Spark materializes and registers this execution-context value before native execution.
    // Lookup still happens at runtime; it is independent of the current input row or batch.
    if let Some(subquery) = expr.downcast_ref::<Subquery>() {
        return Some(ConsumerProperties {
            constant: true,
            has_scalar_struct: false,
            bytes: 0,
            unbounded_output: fixed_width_bytes(&subquery.data_type).is_none(),
        });
    }
    if expr.is::<Column>() {
        return Some(ConsumerProperties {
            constant: false,
            has_scalar_struct: false,
            bytes: 0,
            unbounded_output: false,
        });
    }
    if let Some(expr) = expr.downcast_ref::<ScalarSubqueryConsumer>() {
        return consumer_properties(&expr.inner, input_schema);
    }
    if let Some(expr) = expr.downcast_ref::<NullShortCircuit>() {
        return consumer_properties(expr.inner(), input_schema);
    }
    if let Some(fun) = expr.downcast_ref::<ScalarFunctionExpr>() {
        if fun.fun().signature().volatility != Volatility::Immutable {
            return None;
        }
    } else if !(expr.is::<AtLeastNNonNulls>()
        || expr.is::<ArrayInsert>()
        || expr.is::<GetArrayStructFields>()
        || expr.is::<SparkArraysZipFunc>()
        || expr.is::<NestedPredicate>()
        || expr.is::<NormalizeNestedFloats>()
        || expr.is::<FromJson>()
        || expr.is::<ToCsv>()
        || expr.is::<GetStructField>()
        || expr.is::<CastExpr>()
        || expr.is::<Cast>()
        || expr.is::<BinaryExpr>()
        || expr.is::<CheckedBinaryExpr>()
        || expr.is::<CheckOverflow>()
        || expr.is::<DecimalRescaleCheckOverflow>()
        || expr.is::<NormalizeNaNAndZero>()
        || expr.is::<NegativeExpr>()
        || expr.is::<SparkComparison>()
        || expr.is::<IsNullExpr>()
        || expr.is::<IsNotNullExpr>()
        || expr.is::<NotExpr>()
        || expr.is::<InListExpr>()
        || expr.is::<LikeExpr>()
        || expr.is::<RLike>()
        || expr.is::<IfExpr>()
        || expr.is::<CaseWhenExpr>()
        || expr.is::<CreateNamedStruct>()
        || expr.is::<ListExtract>()
        || expr.is::<ToJson>())
    {
        return None;
    }
    // Bound fixed outputs such as integer formatting and fixed-cardinality constructors.
    // Descendant unbounded intermediates are still propagated below.
    let sizes_output = expr
        .downcast_ref::<ScalarFunctionExpr>()
        .map(ScalarFunctionExpr::return_type)
        .or_else(|| expr.downcast_ref::<Cast>().map(|cast| &cast.data_type))
        .or_else(|| expr.downcast_ref::<CastExpr>().map(CastExpr::cast_type))
        .is_some_and(|dt| {
            matches!(
                dt,
                DataType::Utf8
                    | DataType::LargeUtf8
                    | DataType::Utf8View
                    | DataType::Binary
                    | DataType::LargeBinary
                    | DataType::BinaryView
            ) || dt.is_nested()
        })
        || expr.is::<ToJson>()
        || expr.is::<ToCsv>()
        || expr.is::<FromJson>()
        || expr.is::<ArrayInsert>()
        || expr.is::<SparkArraysZipFunc>()
        || expr
            .downcast_ref::<ListExtract>()
            .is_some_and(|e| e.default_value().is_some());
    let output_bytes = if sizes_output {
        bounded_output_bytes(expr, input_schema)
    } else {
        Some(0)
    };
    let mut properties = ConsumerProperties {
        constant: true,
        has_scalar_struct: false,
        bytes: output_bytes.unwrap_or(0),
        unbounded_output: output_bytes.is_none(),
    };
    let default_value = expr
        .downcast_ref::<ListExtract>()
        .and_then(ListExtract::default_value);
    for child in expr.children().into_iter().chain(default_value) {
        let child = consumer_properties(child, input_schema)?;
        properties.constant &= child.constant;
        properties.has_scalar_struct |= child.has_scalar_struct;
        properties.bytes = properties.bytes.saturating_add(child.bytes);
        properties.unbounded_output |= child.unbounded_output;
    }
    // Membership stops when the entire batch has matched. Only constant membership can
    // change batch boundaries without changing evaluation of later ANSI-error candidates.
    if !properties.constant
        && expr
            .downcast_ref::<NestedPredicate>()
            .is_some_and(NestedPredicate::is_membership)
    {
        return None;
    }
    Some(properties)
}

// Conservative per-row bytes, including one byte per validity mask. Unknown variable-length
// types stay unbounded; a constructor may supply a tighter expression-specific bound.
fn fixed_width_bytes(dt: &DataType) -> Option<usize> {
    let bytes = match dt {
        DataType::Null => 0,
        DataType::Boolean => 1,
        DataType::FixedSizeBinary(n) => usize::try_from(*n).ok()?,
        DataType::FixedSizeList(field, n) => {
            fixed_width_bytes(field.data_type())?.checked_mul(usize::try_from(*n).ok()?)?
        }
        DataType::Struct(fields) => fields.iter().try_fold(0usize, |total, field| {
            total.checked_add(fixed_width_bytes(field.data_type())?)
        })?,
        _ => dt.primitive_width()?,
    };
    bytes.checked_add(1)
}

fn bounded_output_bytes(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Option<usize> {
    if let Some(consumer) = expr.downcast_ref::<ScalarSubqueryConsumer>() {
        return bounded_output_bytes(&consumer.inner, schema);
    }
    if let Some(consumer) = expr.downcast_ref::<NullShortCircuit>() {
        return bounded_output_bytes(consumer.inner(), schema);
    }
    if let Some(field) = expr.downcast_ref::<GetStructField>() {
        if let Some(value) = field.literal_field() {
            return value.to_data().get_slice_memory_size().ok();
        }
    }
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return literal_bytes(literal.value());
    }
    let dt = expr.data_type(schema).ok()?;
    if let Some(bytes) = fixed_width_bytes(&dt) {
        return Some(bytes);
    }
    let choice = expr
        .downcast_ref::<CaseWhenExpr>()
        .or_else(|| expr.downcast_ref::<IfExpr>().map(IfExpr::case_when));
    if let Some(choice) = choice {
        return choice.values().try_fold(0usize, |bound, value| {
            Some(bound.max(bounded_output_bytes(value, schema)?))
        });
    }
    if let Some(fun) = expr.downcast_ref::<ScalarFunctionExpr>() {
        let implementation = fun.fun().inner();
        if implementation.is::<datafusion_functions::core::greatest::GreatestFunc>()
            || implementation.is::<datafusion_functions::core::least::LeastFunc>()
            || implementation.is::<crate::SparkGreatestLeast>()
        {
            return fun.args().iter().try_fold(0usize, |bound, value| {
                Some(bound.max(bounded_output_bytes(value, schema)?))
            });
        }
    }
    None
}

fn literal_bytes(value: &ScalarValue) -> Option<usize> {
    // Counting a scalar payload must not copy it into Arrow at every enclosing expression.
    match value {
        ScalarValue::Utf8(value) => Some(value.as_ref().map_or(0, |v| v.len()).saturating_add(9)),
        ScalarValue::Binary(value) => Some(value.as_ref().map_or(0, |v| v.len()).saturating_add(9)),
        ScalarValue::LargeUtf8(value) => {
            Some(value.as_ref().map_or(0, |v| v.len()).saturating_add(17))
        }
        ScalarValue::LargeBinary(value) => {
            Some(value.as_ref().map_or(0, |v| v.len()).saturating_add(17))
        }
        _ => value
            .to_array()
            .ok()?
            .to_data()
            .get_slice_memory_size()
            .ok(),
    }
}

fn variable_bytes(data: &ArrayData) -> usize {
    match data.data_type() {
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Binary | DataType::LargeBinary => {
            data.buffers()[1].len()
        }
        DataType::List(_)
        | DataType::LargeList(_)
        | DataType::FixedSizeList(_, _)
        | DataType::Map(_, _) => data.get_buffer_memory_size(),
        _ => data
            .child_data()
            .iter()
            .map(variable_bytes)
            .fold(0, usize::saturating_add),
    }
}

impl Hash for ScalarSubqueryConsumer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl PartialEq for ScalarSubqueryConsumer {
    fn eq(&self, other: &Self) -> bool {
        self.inner.eq(&other.inner)
    }
}

impl Display for ScalarSubqueryConsumer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let operation = self
            .inner
            .downcast_ref::<ScalarFunctionExpr>()
            .map_or("expression", ScalarFunctionExpr::name);
        // Whole-struct consumers also reach DataFusion's retained metric labels. Keep the
        // operation and evaluation shape, without formatting owned scalar payloads.
        write!(
            f,
            "ScalarSubqueryConsumer [operation: {operation}, type: {:?}, constant: {}, max_rows: {}]",
            self.inner.data_type(&self.input_schema), self.constant, self.max_rows
        )
    }
}

impl Debug for ScalarSubqueryConsumer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }
}

impl PhysicalExpr for ScalarSubqueryConsumer {
    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        Display::fmt(self, f)
    }

    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        self.inner.data_type(input_schema)
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.inner.nullable(input_schema)
    }

    fn return_field(&self, input_schema: &Schema) -> Result<FieldRef> {
        self.inner.return_field(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
        let rows = batch.num_rows();
        if rows == 0 {
            return Ok(ColumnarValue::Array(new_empty_array(
                &self.data_type(batch.schema_ref())?,
            )));
        }
        if !self.constant {
            if rows <= self.max_rows {
                return self.inner.evaluate(batch);
            }
            let mut arrays: Vec<ArrayRef> = Vec::new();
            for offset in (0..rows).step_by(self.max_rows) {
                let len = self.max_rows.min(rows - offset);
                let array = self
                    .inner
                    .evaluate(&batch.slice(offset, len))?
                    .into_array(len)?;
                if array.len() != len {
                    return internal_err!(
                        "Scalar subquery consumer returned {} rows, expected {len}",
                        array.len()
                    );
                }
                arrays.push(array);
            }
            let arrays: Vec<&dyn Array> = arrays.iter().map(|array| array.as_ref()).collect();
            return Ok(ColumnarValue::Array(concat(&arrays)?));
        }
        let value = match self.inner.evaluate(&batch.slice(0, 1))? {
            ColumnarValue::Scalar(value) => value,
            ColumnarValue::Array(array) => {
                if array.len() != 1 {
                    return internal_err!("Scalar subquery consumer returned {} rows", array.len());
                }
                if rows == 1 {
                    return Ok(ColumnarValue::Array(array));
                }
                ScalarValue::try_from_array(array.as_ref(), 0)?
            }
        };
        // A genuinely oversized final string/binary column cannot use 32-bit Arrow offsets.
        // Report it as an execution error rather than panicking in new_repeated.
        let bytes = match &value {
            ScalarValue::Utf8(Some(value)) => value.len(),
            ScalarValue::Binary(Some(value)) => value.len(),
            _ => 0,
        };
        if bytes
            .checked_mul(rows)
            .is_none_or(|n| n > i32::MAX as usize)
        {
            return exec_err!(
                "Scalar subquery consumer result exceeds Arrow's 32-bit offset limit"
            );
        }
        Ok(ColumnarValue::Array(value.to_array_of_size(rows)?))
    }

    fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
        vec![&self.inner]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn PhysicalExpr>>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        let [inner]: [Arc<dyn PhysicalExpr>; 1] = children.try_into().map_err(|_| {
            datafusion::common::internal_datafusion_err!("Expected one scalar subquery consumer")
        })?;
        // Recheck after rewrites: constant and row-dependent consumers need different evaluation.
        Ok(Self::wrap_with_schema(
            inner,
            Arc::clone(&self.input_schema),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn literal_payload_bounds_include_offsets_and_nulls() {
        for value in [
            ScalarValue::Utf8(Some("λ".repeat(1024))),
            ScalarValue::Utf8(None),
            ScalarValue::LargeUtf8(Some("λ".repeat(1024))),
            ScalarValue::LargeUtf8(None),
            ScalarValue::Binary(Some(vec![7; 4096])),
            ScalarValue::Binary(None),
            ScalarValue::LargeBinary(Some(vec![7; 4096])),
            ScalarValue::LargeBinary(None),
        ] {
            let array = value.to_array().unwrap();
            let actual = array.to_data().get_slice_memory_size().unwrap();
            assert!(literal_bytes(&value).unwrap() >= actual, "{value:?}");
        }
    }
}
