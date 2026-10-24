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
use std::fmt::{Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

/// Evaluates scalar-struct consumers before repeating large variable-width fields.
///
/// GetStructField retains its batch-sized array contract. Constant consumers run on one row;
/// pure row-dependent consumers run on bounded slices before their results are concatenated.
/// Nothing is evaluated during planning, preserving conditional and ANSI NULL short-circuiting.
#[derive(Debug, Eq)]
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

    /// Projection inlining must not move opaque or stateful consumers onto a new batch.
    pub fn supports_sliced_evaluation(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
        consumer_properties(expr, schema).is_some()
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
        let bytes = match literal.value() {
            ScalarValue::Utf8(Some(value)) | ScalarValue::LargeUtf8(Some(value)) => value.len(),
            ScalarValue::Binary(Some(value)) | ScalarValue::LargeBinary(Some(value)) => value.len(),
            ScalarValue::Struct(value) => variable_bytes(&value.to_data()),
            _ => 0,
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
    if expr.is::<Subquery>() {
        return Some(ConsumerProperties {
            constant: true,
            has_scalar_struct: false,
            bytes: 0,
            unbounded_output: false,
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

fn bounded_string_cast_bytes(dt: &DataType) -> Option<usize> {
    match dt {
        DataType::Boolean => Some(5),
        DataType::Decimal128(precision, scale) => {
            let precision = usize::from(*precision);
            let scale = i16::from(*scale);
            // ANSI/TRY and Arrow use plain notation; LEGACY can use Java scientific notation.
            let plain = if scale >= 0 {
                (precision + 2).max(scale as usize + 3)
            } else {
                precision + usize::from(scale.unsigned_abs()) + 1
            };
            Some(plain.max(precision + 7))
        }
        DataType::Int8 => Some(4),
        DataType::Int16 => Some(6),
        DataType::Int32 => Some(11),
        DataType::Int64 => Some(20),
        DataType::UInt8 => Some(3),
        DataType::UInt16 => Some(5),
        DataType::UInt32 => Some(10),
        DataType::UInt64 => Some(20),
        _ => None,
    }
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
        return literal
            .value()
            .to_array()
            .ok()?
            .to_data()
            .get_slice_memory_size()
            .ok();
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
    let cast_child = expr
        .downcast_ref::<Cast>()
        .map(|cast| &cast.child)
        .or_else(|| expr.downcast_ref::<CastExpr>().map(CastExpr::expr));
    if let Some(child) = cast_child {
        if matches!(
            dt,
            DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
        ) {
            // Five bytes cover a Utf8 offset and validity; use nine for LargeUtf8.
            let overhead = match dt {
                DataType::LargeUtf8 => 9,
                DataType::Utf8View => 17,
                _ => 5,
            };
            let child_type = child.data_type(schema).ok()?;
            let bytes = bounded_string_cast_bytes(&child_type).or_else(|| {
                // Comet's Java-style formatter switches to scientific notation outside
                // [1e-3, 1e7). IEEE digits, sign and exponent fit in 32 ASCII bytes.
                // Arrow's CastExpr uses a different formatter, so do not infer its bound.
                (expr.is::<Cast>() && matches!(child_type, DataType::Float32 | DataType::Float64))
                    .then_some(32)
            })?;
            return bytes.checked_add(overhead);
        }
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
        if fun.fun().name() == "make_array" {
            // Arity bounds the list cardinality; each child bounds its one element's bytes.
            return fun.args().iter().try_fold(5usize, |total, child| {
                total.checked_add(bounded_output_bytes(child, schema)?)
            });
        }
    }
    if expr.is::<CreateNamedStruct>() {
        return expr.children().iter().try_fold(1usize, |total, child| {
            total.checked_add(bounded_output_bytes(child, schema)?)
        });
    }
    None
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
        write!(f, "ScalarSubqueryConsumer({})", self.inner)
    }
}

impl PhysicalExpr for ScalarSubqueryConsumer {
    fn fmt_sql(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        self.inner.fmt_sql(f)
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
