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

//! Small Top-K with Spark's comparator evaluation schedule.
//!
//! The planner must admit only deterministic, stateless integer keys whose only possible
//! SQL error is ANSI Add overflow over proven-safe children. Successful evaluations are
//! pure, so a batch cache is valid; an overflow is rethrown only when Spark's comparator
//! would evaluate that row/key. Ordinary SortExec remains responsible for other sorts.
//!
//! The selector mirrors Guava 33.2.1 TopKSelector, including its comparison direction and
//! final sort before truncation. fetch <= 15 means every Java Arrays.sort call has fewer
//! than 32 entries and uses TimSort's run detection and binary insertion, without merges.

use std::cmp::Ordering;
use std::fmt::Formatter;
use std::mem::size_of;
use std::sync::Arc;

use arrow::array::{
    make_array, Array, ArrayData, ArrayRef, AsArray, PrimitiveArray, PrimitiveBuilder, RecordBatch,
    RecordBatchOptions,
};
use arrow::compute::interleave_record_batch;
use arrow::datatypes::{ArrowPrimitiveType, DataType, Int32Type, Int64Type, SchemaRef};
use arrow::row::{RowConverter, Rows, SortField};
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{internal_err, DataFusionError, Result};
use datafusion::execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion::execution::TaskContext;
use datafusion::physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, CardinalityEffect, EmissionType};
use datafusion::physical_plan::metrics::{BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    apply_expression_roots, DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties,
    PlanProperties, SendableRecordBatchStream,
};
use datafusion_comet_common::{SparkError, SparkErrorWithContext};
use futures::StreamExt;

pub const MAX_SPARK_TOPK_FETCH: usize = 15;

#[derive(Debug)]
pub struct SparkTopKExec {
    input: Arc<dyn ExecutionPlan>,
    // Keep duplicates: deduping comparisons can change which failure Spark observes.
    keys: Vec<PhysicalSortExpr>,
    fetch: usize,
    cache: Arc<PlanProperties>,
    metrics: ExecutionPlanMetricsSet,
}

impl SparkTopKExec {
    pub fn try_new(
        input: Arc<dyn ExecutionPlan>,
        keys: Vec<PhysicalSortExpr>,
        fetch: usize,
    ) -> Result<Self> {
        if !(1..=MAX_SPARK_TOPK_FETCH).contains(&fetch) || keys.is_empty() {
            return internal_err!("SparkTopKExec requires 1..=15 rows and at least one key");
        }
        for key in &keys {
            if !matches!(
                key.expr.data_type(&input.schema())?,
                DataType::Int32 | DataType::Int64
            ) {
                return internal_err!("SparkTopKExec requires admitted INT/BIGINT keys");
            }
        }
        let mut equivalence = input.equivalence_properties().clone();
        equivalence.reorder(LexOrdering::new(keys.clone()).expect("nonempty ordering"))?;
        let cache = Arc::new(PlanProperties::new(
            equivalence,
            input.output_partitioning().clone(),
            EmissionType::Final,
            Boundedness::Bounded,
        ));
        Ok(Self {
            input,
            keys,
            fetch,
            cache,
            metrics: ExecutionPlanMetricsSet::new(),
        })
    }
}

impl DisplayAs for SparkTopKExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut Formatter) -> std::fmt::Result {
        write!(
            f,
            "CometSparkTopKExec: fetch={}, keys={:?}",
            self.fetch, self.keys
        )
    }
}

impl ExecutionPlan for SparkTopKExec {
    fn name(&self) -> &str {
        "CometSparkTopKExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.cache
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }
    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
    fn maintains_input_order(&self) -> Vec<bool> {
        vec![false]
    }
    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }
    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(self.keys.iter().map(|key| &key.expr), f)
    }
    fn with_new_children(
        self: Arc<Self>,
        mut children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if children.len() != 1 {
            return internal_err!("SparkTopKExec requires one child");
        }
        Ok(Arc::new(Self::try_new(
            children.remove(0),
            self.keys.clone(),
            self.fetch,
        )?))
    }
    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let input = self.input.execute(partition, Arc::clone(&context))?;
        let schema = self.schema();
        let encoder = KeyEncoder::try_new(&self.keys, &schema)?;
        let reservation = MemoryConsumer::new(format!("CometSparkTopKExec[{partition}]"))
            .register(&context.runtime_env().memory_pool);
        let metrics = BaselineMetrics::new(&self.metrics, partition);
        let fetch = self.fetch;
        let output_schema = Arc::clone(&schema);
        let stream = futures::stream::once(async move {
            select(input, output_schema, encoder, fetch, reservation, metrics).await
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(schema, stream)))
    }
}

struct KeyEncoder {
    expressions: Vec<Arc<dyn PhysicalExpr>>,
    converters: Vec<RowConverter>,
}

struct KeyCache {
    rows: Rows,
    // A byte vector has a predictable allocation; it records only expected Add errors.
    failed: Vec<u8>,
}

struct CachedBatch {
    batch: RecordBatch,
    keys: Vec<KeyCache>,
    // Zero for a slice whose complete input buffers are already reserved by select.
    payload_bytes: usize,
}

#[derive(Clone)]
struct Candidate {
    batch: Arc<CachedBatch>,
    row: usize,
}

impl Candidate {
    fn same_row(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.batch, &other.batch) && self.row == other.row
    }
}

impl CachedBatch {
    fn size(&self) -> usize {
        self.payload_bytes
            + size_of::<Self>()
            + self.keys.capacity() * size_of::<KeyCache>()
            + self
                .keys
                .iter()
                .map(|key| key.rows.size() + key.failed.capacity())
                .sum::<usize>()
    }
}

impl KeyEncoder {
    fn try_new(keys: &[PhysicalSortExpr], schema: &SchemaRef) -> Result<Self> {
        let converters = keys
            .iter()
            .map(|key| {
                Ok(RowConverter::new(vec![SortField::new_with_options(
                    key.expr.data_type(schema)?,
                    key.options,
                )])?)
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            expressions: keys.iter().map(|key| Arc::clone(&key.expr)).collect(),
            converters,
        })
    }

    fn size(&self) -> usize {
        self.converters
            .iter()
            .map(RowConverter::size)
            .sum::<usize>()
            + self.expressions.capacity() * size_of::<Arc<dyn PhysicalExpr>>()
    }

    fn encode(&self, batch: RecordBatch) -> Result<CachedBatch> {
        let mut keys = Vec::with_capacity(self.expressions.len());
        for (expr, converter) in self.expressions.iter().zip(&self.converters) {
            let n = batch.num_rows();
            let (values, failed) = match expr.evaluate(&batch).and_then(|v| v.into_array(n)) {
                Ok(values) => (values, vec![0; n]),
                Err(error) if is_add_overflow(&error) => match expr.data_type(&batch.schema())? {
                    DataType::Int32 => evaluate_rows::<Int32Type>(expr, &batch)?,
                    DataType::Int64 => evaluate_rows::<Int64Type>(expr, &batch)?,
                    _ => return internal_err!("Unexpected SparkTopKExec key type"),
                },
                Err(error) => return Err(error),
            };
            if values.len() != n {
                return internal_err!("SparkTopKExec key length differs from input");
            }
            keys.push(KeyCache {
                rows: converter.convert_columns(&[values])?,
                failed,
            });
        }
        let payload_bytes = batch.get_array_memory_size();
        Ok(CachedBatch {
            batch,
            keys,
            payload_bytes,
        })
    }

    fn compare(&self, left: &Candidate, right: &Candidate) -> Result<Ordering> {
        for (key, expr) in self.expressions.iter().enumerate() {
            // Spark evaluates the right key even when the left key is NULL. Neither row
            // identity nor a cached NULL is permission to bypass either evaluation.
            replay_failure(expr, left, key)?;
            replay_failure(expr, right, key)?;
            let order = left.batch.keys[key]
                .rows
                .row(left.row)
                .cmp(&right.batch.keys[key].rows.row(right.row));
            if order != Ordering::Equal {
                return Ok(order);
            }
        }
        Ok(Ordering::Equal)
    }
}

fn is_add_overflow(error: &DataFusionError) -> bool {
    let DataFusionError::External(error) = error.find_root() else {
        return false;
    };
    let spark = error.downcast_ref::<SparkError>().or_else(|| {
        error
            .downcast_ref::<SparkErrorWithContext>()
            .map(|e| &e.error)
    });
    matches!(
        spark,
        Some(SparkError::ArithmeticOverflow { .. } | SparkError::BinaryArithmeticOverflow { .. })
    )
}

fn evaluate_rows<T: ArrowPrimitiveType>(
    expr: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
) -> Result<(ArrayRef, Vec<u8>)> {
    let mut values = PrimitiveBuilder::<T>::with_capacity(batch.num_rows());
    let mut failed = Vec::with_capacity(batch.num_rows());
    for row in 0..batch.num_rows() {
        match expr
            .evaluate(&batch.slice(row, 1))
            .and_then(|v| v.into_array(1))
        {
            Ok(value) => {
                let value = value
                    .as_any()
                    .downcast_ref::<PrimitiveArray<T>>()
                    .ok_or_else(|| {
                        DataFusionError::Internal("Unexpected SparkTopKExec row key type".into())
                    })?;
                if value.len() != 1 {
                    return internal_err!("SparkTopKExec row key length differs from one");
                }
                values.append_option((!value.is_null(0)).then(|| value.value(0)));
                failed.push(0);
            }
            Err(error) if is_add_overflow(&error) => {
                values.append_null();
                failed.push(1);
            }
            Err(error) => return Err(error),
        }
    }
    Ok((Arc::new(values.finish()), failed))
}

fn replay_failure(expr: &Arc<dyn PhysicalExpr>, row: &Candidate, key: usize) -> Result<()> {
    if row.batch.keys[key].failed[row.row] != 0 {
        match expr.evaluate(&row.batch.batch.slice(row.row, 1)) {
            Err(error) => return Err(error),
            Ok(_) => return internal_err!("SparkTopKExec admitted a non-repeatable key"),
        }
    }
    Ok(())
}

fn checked_bytes(a: usize, b: usize) -> Result<usize> {
    a.checked_add(b).ok_or_else(|| {
        DataFusionError::ResourcesExhausted("SparkTopKExec memory size overflow".into())
    })
}

fn key_scratch_bytes(rows: usize, keys: usize) -> Result<usize> {
    // Per fixed-width key: output array, row bytes/offsets, failure flags, and one-row
    // retry/builder scratch. This is a conservative preallocation charge, replaced
    // with actual retained capacities after encoding.
    rows.checked_mul(64)
        .and_then(|n| n.checked_mul(keys))
        .and_then(|n| n.checked_add(keys.saturating_mul(1024)))
        .ok_or_else(|| {
            DataFusionError::ResourcesExhausted("SparkTopKExec key size overflow".into())
        })
}

fn reserve_key_rows(
    reservation: &MemoryReservation,
    base: usize,
    mut count: usize,
    keys: usize,
) -> Result<usize> {
    loop {
        let bytes = checked_bytes(base, key_scratch_bytes(count, keys)?)?;
        match reservation.try_resize(bytes) {
            Ok(()) => return Ok(count),
            Err(error)
                if count > 1
                    && matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)) =>
            {
                count = count.div_ceil(2);
            }
            Err(error) => return Err(error),
        }
    }
}

// Bound selected-value copies and interleave's child-index workspace. Arrow's
// generic slice size includes an entire list/map child, which needs slicing too.
fn copied_value_bytes(data: &ArrayData) -> Result<usize> {
    let nested = match data.data_type() {
        DataType::Struct(_) => Some((0, data.len(), 0)),
        DataType::List(_) | DataType::Map(_, _) => {
            let offsets = data.buffer::<i32>(0);
            Some((
                offsets[0] as usize,
                (offsets[data.len()] - offsets[0]) as usize,
                4,
            ))
        }
        DataType::LargeList(_) => {
            let offsets = data.buffer::<i64>(0);
            Some((
                offsets[0] as usize,
                (offsets[data.len()] - offsets[0]) as usize,
                8,
            ))
        }
        DataType::FixedSizeList(_, width) => Some((
            data.offset() * *width as usize,
            data.len() * *width as usize,
            0,
        )),
        _ => None,
    };
    let bytes = if let Some((start, len, offset_width)) = nested {
        let mut size = checked_bytes(data.len().div_ceil(8), (data.len() + 1) * offset_width)?;
        for child in data.child_data() {
            let child = if matches!(data.data_type(), DataType::Struct(_)) {
                child.clone()
            } else {
                child.slice(start, len)
            };
            size = checked_bytes(size, copied_value_bytes(&child)?)?;
        }
        checked_bytes(size, len.saturating_mul(2 * size_of::<usize>()))?
    } else if matches!(data.data_type(), DataType::Utf8View | DataType::BinaryView) {
        let array = make_array(data.clone());
        let values = if matches!(data.data_type(), DataType::Utf8View) {
            array.as_string_view().total_buffer_bytes_used()
        } else {
            array.as_binary_view().total_buffer_bytes_used()
        };
        checked_bytes(data.len().saturating_mul(16), values)?
    } else if matches!(data.data_type(), DataType::Dictionary(_, _)) {
        // Dictionary interleave retains its dictionary values.
        data.get_buffer_memory_size()
    } else {
        data.get_slice_memory_size()?
    };
    checked_bytes(bytes.saturating_mul(2), 256 + size_of::<ArrayData>())
}

fn compact_view_buffers(data: ArrayData) -> Result<ArrayData> {
    match data.data_type() {
        DataType::Utf8View => Ok(make_array(data).as_string_view().gc().to_data()),
        DataType::BinaryView => Ok(make_array(data).as_binary_view().gc().to_data()),
        DataType::Dictionary(_, _) => Ok(data),
        _ if data.child_data().is_empty() => Ok(data),
        _ => {
            let children = data
                .child_data()
                .iter()
                .cloned()
                .map(compact_view_buffers)
                .collect::<Result<Vec<_>>>()?;
            Ok(data.into_builder().child_data(children).build()?)
        }
    }
}

fn retained_batches(selector: &Selector<Candidate>) -> Vec<Arc<CachedBatch>> {
    let mut batches = Vec::<Arc<CachedBatch>>::new();
    for candidate in &selector.buffer {
        if !batches
            .iter()
            .any(|batch| Arc::ptr_eq(batch, &candidate.batch))
        {
            batches.push(Arc::clone(&candidate.batch));
        }
    }
    batches
}

fn retained_size(selector: &Selector<Candidate>) -> Result<usize> {
    retained_batches(selector)
        .iter()
        .try_fold(0, |size, batch| checked_bytes(size, batch.size()))
}

/// Copy only the at-most-2k candidate payloads between batches. No previous full input
/// batch is pinned while asking the child for its next batch.
fn compact(
    selector: &mut Selector<Candidate>,
    encoder: &KeyEncoder,
    schema: &SchemaRef,
    reservation: &MemoryReservation,
    fixed: usize,
) -> Result<()> {
    if selector.buffer.is_empty() {
        return Ok(());
    }
    let threshold_position = selector
        .threshold
        .as_ref()
        .and_then(|threshold| {
            selector
                .buffer
                .iter()
                .position(|row| row.same_row(threshold))
        })
        .ok_or_else(|| {
            DataFusionError::Internal("SparkTopKExec threshold left its buffer".into())
        })?;
    let sources = retained_batches(selector);
    let copy_bytes = selector.buffer.iter().try_fold(0, |size, row| {
        row.batch
            .batch
            .columns()
            .iter()
            .try_fold(size, |size, column| {
                checked_bytes(
                    size,
                    copied_value_bytes(&column.to_data().slice(row.row, 1))?,
                )
            })
    })?;
    // Charge selected values while both old and compacted storage are alive.
    let temporary = checked_bytes(
        copy_bytes,
        key_scratch_bytes(selector.buffer.len(), encoder.expressions.len())?,
    )?;
    reservation.try_resize(checked_bytes(
        checked_bytes(fixed, retained_size(selector)?)?,
        temporary,
    )?)?;
    let indices = selector
        .buffer
        .iter()
        .map(|row| {
            (
                sources
                    .iter()
                    .position(|source| Arc::ptr_eq(source, &row.batch))
                    .expect("retained batch"),
                row.row,
            )
        })
        .collect::<Vec<_>>();
    let batches = sources
        .iter()
        .map(|source| &source.batch)
        .collect::<Vec<_>>();
    let batch = if schema.fields().is_empty() {
        RecordBatch::try_new_with_options(
            Arc::clone(schema),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(indices.len())),
        )?
    } else {
        let batch = interleave_record_batch(&batches, &indices)?;
        let columns = batch
            .columns()
            .iter()
            .map(|column| compact_view_buffers(column.to_data()).map(make_array))
            .collect::<Result<Vec<_>>>()?;
        RecordBatch::try_new(Arc::clone(schema), columns)?
    };
    let mut keys = Vec::with_capacity(encoder.expressions.len());
    for (key, converter) in encoder.converters.iter().enumerate() {
        let data_bytes = selector
            .buffer
            .iter()
            .map(|row| row.batch.keys[key].rows.row(row.row).as_ref().len())
            .sum();
        let mut rows = converter.empty_rows(selector.buffer.len(), data_bytes);
        let mut failed = Vec::with_capacity(selector.buffer.len());
        for row in &selector.buffer {
            rows.push(row.batch.keys[key].rows.row(row.row));
            failed.push(row.batch.keys[key].failed[row.row]);
        }
        keys.push(KeyCache { rows, failed });
    }
    let payload_bytes = batch.get_array_memory_size();
    let compacted = Arc::new(CachedBatch {
        batch,
        keys,
        payload_bytes,
    });
    for (index, row) in selector.buffer.iter_mut().enumerate() {
        *row = Candidate {
            batch: Arc::clone(&compacted),
            row: index,
        };
    }
    selector.threshold = Some(selector.buffer[threshold_position].clone());
    drop(sources);
    reservation.try_resize(checked_bytes(fixed, compacted.size())?)?;
    Ok(())
}

async fn select(
    mut input: SendableRecordBatchStream,
    schema: SchemaRef,
    encoder: KeyEncoder,
    fetch: usize,
    reservation: MemoryReservation,
    metrics: BaselineMetrics,
) -> Result<RecordBatch> {
    let mut selector = Selector::new(fetch);
    let fixed = encoder.size()
        + selector.buffer.capacity() * size_of::<Candidate>()
        + size_of::<Selector<Candidate>>();
    reservation.try_resize(fixed)?;
    while let Some(batch) = input.next().await {
        let batch = batch?;
        if batch.num_rows() == 0 {
            continue;
        }
        let _timer = metrics.elapsed_compute().timer();
        let held_input = checked_bytes(fixed, batch.get_array_memory_size())?;
        let mut offset = 0;
        while offset < batch.num_rows() {
            let base = checked_bytes(held_input, retained_size(&selector)?)?;
            // Key caches are optional workspace, not a minimum input-batch-sized
            // allocation. Retrying smaller slices preserves every comparator call.
            let count = reserve_key_rows(
                &reservation,
                base,
                batch.num_rows() - offset,
                encoder.expressions.len(),
            )?;
            let mut cached = encoder.encode(batch.slice(offset, count))?;
            cached.payload_bytes = 0;
            let cached = Arc::new(cached);
            reservation.try_resize(checked_bytes(base, cached.size())?)?;
            let mut candidate = Candidate {
                batch: Arc::clone(&cached),
                row: 0,
            };
            for row in 0..count {
                candidate.row = row;
                selector.offer_ref(&candidate, &mut |a, b| encoder.compare(a, b))?;
            }
            drop(candidate);
            drop(cached);
            compact(&mut selector, &encoder, &schema, &reservation, held_input)?;
            offset += count;
        }
        drop(batch);
        reservation.try_resize(checked_bytes(fixed, retained_size(&selector)?)?)?;
    }
    let _timer = metrics.elapsed_compute().timer();
    selector.finish(&mut |a, b| encoder.compare(a, b))?;
    compact(&mut selector, &encoder, &schema, &reservation, fixed)?;
    let output = selector
        .buffer
        .first()
        .map(|row| row.batch.batch.clone())
        .unwrap_or_else(|| RecordBatch::new_empty(schema));
    metrics.record_output(output.num_rows());
    Ok(output)
}

/// Merge two already-sorted, at-most-fetch runs for Spark's driver collect path.
/// The caller keeps the original payload rows; only referenced INT/BIGINT attributes
/// need to be transported. This synchronous bridge retains no state after it returns.
pub fn merge_indices(
    batch: &RecordBatch,
    split: usize,
    fetch: usize,
    keys: &[PhysicalSortExpr],
) -> Result<Vec<usize>> {
    if !(1..=MAX_SPARK_TOPK_FETCH).contains(&fetch)
        || split > batch.num_rows()
        || split > fetch
        || batch.num_rows() - split > fetch
        || keys.is_empty()
    {
        return internal_err!("Spark Top-K merge requires two runs of at most fetch <= 15");
    }
    for key in keys {
        if !matches!(
            key.expr.data_type(&batch.schema())?,
            DataType::Int32 | DataType::Int64
        ) {
            return internal_err!("Spark Top-K merge requires admitted INT/BIGINT keys");
        }
    }
    let encoder = KeyEncoder::try_new(keys, &batch.schema())?;
    let cached = Arc::new(encoder.encode(batch.clone())?);
    merge_runs(split, batch.num_rows(), fetch, &mut |left, right| {
        encoder.compare(
            &Candidate {
                batch: Arc::clone(&cached),
                row: left,
            },
            &Candidate {
                batch: Arc::clone(&cached),
                row: right,
            },
        )
    })
}

/// Guava Iterators.MergingIterator with two PeekingIterators. PriorityQueue.offer
/// compares the newly inserted head against the other head and moves it only on <0.
/// Reinsertion occurs inside next(), before the returned row reaches take(fetch).
fn merge_runs<F: FnMut(usize, usize) -> Result<Ordering>>(
    split: usize,
    rows: usize,
    fetch: usize,
    compare: &mut F,
) -> Result<Vec<usize>> {
    let mut left = 0;
    let mut right = split;
    let mut left_first = if left == split {
        false
    } else if right == rows {
        true
    } else {
        compare(right, left)? != Ordering::Less
    };
    let mut result = Vec::with_capacity(fetch.min(rows));
    while result.len() < fetch && (left < split || right < rows) {
        let selected = if left_first { left } else { right };
        if left_first {
            left += 1;
        } else {
            right += 1;
        }
        left_first = if left == split {
            false
        } else if right == rows {
            true
        } else if left_first {
            compare(left, right)? == Ordering::Less
        } else {
            compare(right, left)? != Ordering::Less
        };
        result.push(selected);
    }
    Ok(result)
}

/// Guava's smallest selector. Error propagation deliberately stops the input loop at the
/// exact comparator call; this is not interchangeable with a heap or Rust's stable sort.
struct Selector<T> {
    k: usize,
    buffer: Vec<T>,
    threshold: Option<T>,
}

impl<T: Clone> Selector<T> {
    fn new(k: usize) -> Self {
        assert!((1..=MAX_SPARK_TOPK_FETCH).contains(&k));
        Self {
            k,
            buffer: Vec::with_capacity(2 * k),
            threshold: None,
        }
    }

    #[cfg(test)]
    fn offer<F: FnMut(&T, &T) -> Result<Ordering>>(
        &mut self,
        value: T,
        compare: &mut F,
    ) -> Result<()> {
        self.offer_ref(&value, compare)
    }

    fn offer_ref<F: FnMut(&T, &T) -> Result<Ordering>>(
        &mut self,
        value: &T,
        compare: &mut F,
    ) -> Result<()> {
        if self.buffer.is_empty() {
            self.threshold = Some(value.clone());
            self.buffer.push(value.clone());
        } else if self.buffer.len() < self.k {
            self.buffer.push(value.clone());
            if compare(value, self.threshold.as_ref().expect("nonempty buffer"))?
                == Ordering::Greater
            {
                self.threshold = Some(value.clone());
            }
        } else if compare(value, self.threshold.as_ref().expect("nonempty buffer"))?
            == Ordering::Less
        {
            self.buffer.push(value.clone());
            if self.buffer.len() == 2 * self.k {
                self.trim(compare)?;
            }
        }
        Ok(())
    }

    fn trim<F: FnMut(&T, &T) -> Result<Ordering>>(&mut self, compare: &mut F) -> Result<()> {
        let mut left = 0;
        let mut right = 2 * self.k - 1;
        let mut min_threshold = 0;
        let max_iterations = (usize::BITS - (right - 1).leading_zeros()) * 3;
        let mut iterations = 0;
        while left < right {
            let pivot = (left + right).div_ceil(2);
            let pivot_value = self.buffer[pivot].clone();
            self.buffer[pivot] = self.buffer[right].clone();
            let mut next = left;
            for i in left..right {
                if compare(&self.buffer[i], &pivot_value)? == Ordering::Less {
                    self.buffer.swap(next, i);
                    next += 1;
                }
            }
            self.buffer[right] = self.buffer[next].clone();
            self.buffer[next] = pivot_value;
            if next > self.k {
                right = next - 1;
            } else if next < self.k {
                left = next.max(left + 1);
                min_threshold = next;
            } else {
                break;
            }
            iterations += 1;
            if iterations >= max_iterations {
                java_small_sort(&mut self.buffer[left..=right], compare)?;
                break;
            }
        }
        self.buffer.truncate(self.k);
        let mut threshold = self.buffer[min_threshold].clone();
        for value in &self.buffer[min_threshold + 1..self.k] {
            if compare(value, &threshold)? == Ordering::Greater {
                threshold = value.clone();
            }
        }
        self.threshold = Some(threshold);
        Ok(())
    }

    fn finish<F: FnMut(&T, &T) -> Result<Ordering>>(&mut self, compare: &mut F) -> Result<()> {
        java_small_sort(&mut self.buffer, compare)?;
        if self.buffer.len() > self.k {
            self.buffer.truncate(self.k);
            self.threshold = self.buffer.last().cloned();
        }
        Ok(())
    }
}

/// OpenJDK TimSort.sort when length < MIN_MERGE (32). A descending run is strict;
/// binary insertion inserts ties after existing entries, matching Java's stable order.
fn java_small_sort<T: Clone, F: FnMut(&T, &T) -> Result<Ordering>>(
    values: &mut [T],
    compare: &mut F,
) -> Result<()> {
    if values.len() < 2 {
        return Ok(());
    }
    debug_assert!(values.len() < 32);
    let mut run = 2;
    if compare(&values[1], &values[0])? == Ordering::Less {
        while run < values.len() && compare(&values[run], &values[run - 1])? == Ordering::Less {
            run += 1;
        }
        values[..run].reverse();
    } else {
        while run < values.len() && compare(&values[run], &values[run - 1])? != Ordering::Less {
            run += 1;
        }
    }
    for start in run..values.len() {
        let pivot = values[start].clone();
        let mut left = 0;
        let mut right = start;
        while left < right {
            let mid = (left + right) / 2;
            if compare(&pivot, &values[mid])? == Ordering::Less {
                right = mid;
            } else {
                left = mid + 1;
            }
        }
        for i in (left..start).rev() {
            values[i + 1] = values[i].clone();
        }
        values[left] = pivot;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Int32Array, Int64Array, ListArray, StringArray, StringViewArray};
    use arrow::datatypes::{Field, Schema};
    use datafusion::common::{config::ConfigOptions, ScalarValue};
    use datafusion::execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion::logical_expr::{create_udf, Volatility};
    use datafusion::physical_expr::expressions::{Column, Literal};
    use datafusion::physical_expr::ScalarFunctionExpr;
    use datafusion_comet_spark_expr::{checked_add, CheckedBinaryExpr, EvalMode};

    fn add_key(column: usize, dt: DataType) -> PhysicalSortExpr {
        let one = match dt {
            DataType::Int32 => ScalarValue::Int32(Some(1)),
            DataType::Int64 => ScalarValue::Int64(Some(1)),
            _ => unreachable!(),
        };
        let output = dt.clone();
        let child = Arc::new(ScalarFunctionExpr::new(
            "spark_topk_test_add",
            Arc::new(create_udf(
                "spark_topk_test_add",
                vec![dt.clone(), dt.clone()],
                dt.clone(),
                Volatility::Immutable,
                Arc::new(move |args| checked_add(args, &output, EvalMode::Ansi)),
            )),
            vec![
                Arc::new(Column::new("value", column)),
                Arc::new(Literal::new(one)),
            ],
            Arc::new(Field::new("add", dt, true)),
            Arc::new(ConfigOptions::default()),
        ));
        PhysicalSortExpr::new_default(Arc::new(CheckedBinaryExpr::new(child, None)))
    }

    #[test]
    fn checked_add_overflow_is_cached_but_singleton_never_replays_it() {
        for (dt, values) in [
            (
                DataType::Int32,
                Arc::new(Int32Array::from(vec![i32::MAX])) as ArrayRef,
            ),
            (
                DataType::Int64,
                Arc::new(Int64Array::from(vec![i64::MAX])) as ArrayRef,
            ),
        ] {
            let schema = Arc::new(Schema::new(vec![Field::new("value", dt.clone(), true)]));
            let encoder = KeyEncoder::try_new(&[add_key(0, dt)], &schema).unwrap();
            let batch = Arc::new(
                encoder
                    .encode(RecordBatch::try_new(schema, vec![values]).unwrap())
                    .unwrap(),
            );
            assert_eq!(batch.keys[0].failed, [1]);
            let row = Candidate { batch, row: 0 };
            let mut selector = Selector::new(1);
            selector
                .offer(row.clone(), &mut |a, b| encoder.compare(a, b))
                .unwrap();
            selector.finish(&mut |a, b| encoder.compare(a, b)).unwrap();
            assert!(is_add_overflow(&encoder.compare(&row, &row).unwrap_err()));
        }
    }

    #[test]
    fn checked_secondary_add_stays_lazy_but_null_left_does_not_hide_right_error() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("primary", DataType::Int64, true),
            Field::new("secondary", DataType::Int64, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![0, 1, 0])),
                Arc::new(Int64Array::from(vec![Some(i64::MAX), Some(i64::MAX), None])),
            ],
        )
        .unwrap();
        let encoder = KeyEncoder::try_new(
            &[
                PhysicalSortExpr::new_default(Arc::new(Column::new("primary", 0))),
                add_key(1, DataType::Int64),
            ],
            &schema,
        )
        .unwrap();
        let batch = Arc::new(encoder.encode(batch).unwrap());
        assert_eq!(batch.keys[1].failed, [1, 1, 0]);
        let row = |row| Candidate {
            batch: Arc::clone(&batch),
            row,
        };
        assert_eq!(
            encoder.compare(&row(1), &row(0)).unwrap(),
            Ordering::Greater
        );
        assert!(is_add_overflow(
            &encoder.compare(&row(2), &row(0)).unwrap_err()
        ));
        assert!(is_add_overflow(
            &encoder.compare(&row(0), &row(2)).unwrap_err()
        ));
    }

    #[test]
    fn cache_defers_only_typed_arithmetic_errors() {
        let error = SparkError::ArithmeticOverflow {
            from_type: "long".into(),
            function_name: "try_add".into(),
        };
        assert!(is_add_overflow(&DataFusionError::from(error.clone())));
        assert!(is_add_overflow(&DataFusionError::from(
            SparkErrorWithContext::new(error)
        )));
        assert!(!is_add_overflow(&DataFusionError::from(
            SparkError::DivideByZero
        )));
        assert!(!is_add_overflow(&DataFusionError::ResourcesExhausted(
            "memory".into()
        )));
        assert!(!is_add_overflow(&DataFusionError::Internal(
            "internal".into()
        )));
    }

    #[test]
    fn compaction_releases_full_input_and_reserves_retained_payload_and_keys() {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int64,
            false,
        )]));
        let encoder = KeyEncoder::try_new(&[add_key(0, DataType::Int64)], &schema).unwrap();
        let batch = Arc::new(
            encoder
                .encode(
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(Int64Array::from_iter_values(0..8192))],
                    )
                    .unwrap(),
                )
                .unwrap(),
        );
        let old = Arc::downgrade(&batch);
        let mut selector = Selector::new(2);
        for row in 0..batch.batch.num_rows() {
            selector
                .offer(
                    Candidate {
                        batch: Arc::clone(&batch),
                        row,
                    },
                    &mut |a, b| encoder.compare(a, b),
                )
                .unwrap();
        }
        drop(batch);
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(4 * 1024 * 1024));
        let reservation = MemoryConsumer::new("topk test").register(&pool);
        compact(&mut selector, &encoder, &schema, &reservation, 0).unwrap();
        assert!(old.upgrade().is_none());
        assert_eq!(selector.buffer[0].batch.batch.num_rows(), 2);
        assert_eq!(reservation.size(), selector.buffer[0].batch.size());
        assert_eq!(pool.reserved(), reservation.size());
        assert_eq!(
            encoder
                .compare(&selector.buffer[0], &selector.buffer[1])
                .unwrap(),
            Ordering::Less
        );
        drop(selector);
        drop(reservation);
        assert_eq!(pool.reserved(), 0);
    }

    #[tokio::test]
    async fn key_workspace_adapts_to_a_small_pool_without_changing_selection() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("value", DataType::Int32, false),
            Field::new("payload", DataType::Int32, false),
        ]));
        for count in [1, 2] {
            let batches = (0..2)
                .map(|batch| {
                    RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![
                            Arc::new(Int32Array::from_iter_values(
                                (batch * 8192..(batch + 1) * 8192).rev(),
                            )),
                            Arc::new(Int32Array::from_iter_values(0..8192)),
                        ],
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>();
            let input = Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&schema),
                futures::stream::iter(batches.into_iter().map(Ok)),
            ));
            let keys = vec![add_key(0, DataType::Int32); count];
            let encoder = KeyEncoder::try_new(&keys, &schema).unwrap();
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(512 * 1024));
            let reservation = MemoryConsumer::new("small topk test").register(&pool);
            let metrics = BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
            let output = select(input, Arc::clone(&schema), encoder, 7, reservation, metrics)
                .await
                .unwrap();
            let values = output
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            assert_eq!(values.values().as_ref(), &[0, 1, 2, 3, 4, 5, 6]);
            assert_eq!(pool.reserved(), 0);
        }
    }

    #[tokio::test]
    async fn compaction_copies_selected_wide_and_nested_values_only() {
        for payload in [
            Arc::new(StringArray::from_iter_values(
                (0..8192).map(|i| format!("{i:0256}")),
            )) as ArrayRef,
            Arc::new(StringViewArray::from_iter_values((0..8192).map(|i| format!("{i:0256}"))).gc())
                as ArrayRef,
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                (0..8192).map(|i| Some((0..64).map(move |j| Some(i + j)))),
            )) as ArrayRef,
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("value", DataType::Int32, false),
                Field::new("payload", payload.data_type().clone(), true),
            ]));
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from_iter_values(0..8192)), payload],
            )
            .unwrap();
            let input = Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&schema),
                futures::stream::iter(vec![Ok(batch.clone()), Ok(batch.clone())]),
            ));
            let encoder = KeyEncoder::try_new(&[add_key(0, DataType::Int32)], &schema).unwrap();
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(3 * 1024 * 1024));
            let reservation = MemoryConsumer::new("wide topk test").register(&pool);
            let metrics = BaselineMetrics::new(&ExecutionPlanMetricsSet::new(), 0);
            let output = select(input, schema, encoder, 7, reservation, metrics)
                .await
                .unwrap();
            assert_eq!(
                output
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .as_ref(),
                &[0, 0, 1, 1, 2, 2, 3]
            );
            assert!(output.get_array_memory_size() < 64 * 1024);
            assert_eq!(pool.reserved(), 0);
        }
    }

    #[test]
    fn key_workspace_does_not_retry_an_acquisition_exception() {
        use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};

        #[derive(Debug)]
        struct FailedAcquisition {
            calls: AtomicUsize,
            inner: GreedyMemoryPool,
        }
        impl std::fmt::Display for FailedAcquisition {
            fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
                write!(f, "failed acquisition")
            }
        }
        impl MemoryPool for FailedAcquisition {
            fn name(&self) -> &str {
                "failed acquisition"
            }
            fn grow(&self, r: &MemoryReservation, n: usize) {
                self.inner.grow(r, n)
            }
            fn shrink(&self, r: &MemoryReservation, n: usize) {
                self.inner.shrink(r, n)
            }
            fn reserved(&self) -> usize {
                self.inner.reserved()
            }
            fn try_grow(&self, r: &MemoryReservation, n: usize) -> Result<()> {
                if self.calls.fetch_add(1, AtomicOrdering::SeqCst) == 0 {
                    return Err(DataFusionError::Execution("acquisition exception".into()));
                }
                self.inner.try_grow(r, n)
            }
        }
        let pool = Arc::new(FailedAcquisition {
            calls: AtomicUsize::new(0),
            inner: GreedyMemoryPool::new(1024 * 1024),
        });
        let memory: Arc<dyn MemoryPool> = pool.clone();
        let reservation = MemoryConsumer::new("exception topk test").register(&memory);
        let error = reserve_key_rows(&reservation, 0, 8192, 1).unwrap_err();
        assert!(error.to_string().contains("acquisition exception"));
        assert_eq!(pool.calls.load(AtomicOrdering::SeqCst), 1);
        assert_eq!(pool.reserved(), 0);
    }

    fn failure(name: &str) -> DataFusionError {
        DataFusionError::Execution(name.into())
    }

    #[test]
    fn singleton_never_calls_the_fallible_comparator() {
        let mut selector = Selector::new(1);
        let mut compare = |_: &i32, _: &i32| -> Result<Ordering> { Err(failure("must stay lazy")) };
        selector.offer(7, &mut compare).unwrap();
        selector.finish(&mut compare).unwrap();
        assert_eq!(selector.buffer, vec![7]);
    }

    #[test]
    fn secondary_key_is_lazy_and_left_error_precedes_right_error() {
        let mut seen = Vec::new();
        let mut compare = |a: &(i32, &'static str), b: &(i32, &'static str)| -> Result<Ordering> {
            let primary = a.0.cmp(&b.0);
            if primary != Ordering::Equal {
                return Ok(primary);
            }
            seen.push(a.1);
            if a.1 != "ok" {
                return Err(failure(a.1));
            }
            seen.push(b.1);
            if b.1 != "ok" {
                return Err(failure(b.1));
            }
            Ok(Ordering::Equal)
        };
        let mut selector = Selector::new(1);
        selector.offer((0, "right"), &mut compare).unwrap();
        selector.offer((1, "skipped"), &mut compare).unwrap();
        let error = selector.offer((0, "left"), &mut compare).unwrap_err();
        assert!(error.to_string().contains("left"));
        assert_eq!(seen, vec!["left"]);
    }

    #[test]
    fn offer_error_stops_before_next_input() {
        let mut reads = 0;
        let mut selector = Selector::new(2);
        let result = (|| -> Result<()> {
            for value in [1, 2, 3] {
                reads += 1;
                selector.offer(value, &mut |a, _| {
                    if *a == 2 {
                        Err(failure("key"))
                    } else {
                        Ok(Ordering::Less)
                    }
                })?;
            }
            Ok(())
        })();
        assert!(result.unwrap_err().to_string().contains("key"));
        assert_eq!(reads, 2);
    }

    #[test]
    fn next_input_error_precedes_singleton_sort() {
        let mut selector = Selector::new(2);
        let mut compared = false;
        let mut compare = |_: &i32, _: &i32| -> Result<Ordering> {
            compared = true;
            Err(failure("key"))
        };
        let result = (|| -> Result<()> {
            for next in [Ok(1), Err(failure("input"))] {
                selector.offer(next?, &mut compare)?;
            }
            selector.finish(&mut compare)
        })();
        assert!(result.unwrap_err().to_string().contains("input"));
        assert!(!compared);
    }

    #[test]
    fn selector_keeps_sorted_smallest_values_through_trims() {
        for k in 1..=MAX_SPARK_TOPK_FETCH {
            for input in [
                (0..97).collect::<Vec<i32>>(),
                (0..97).rev().collect(),
                (0..97).map(|i| (i * 37) % 23).collect(),
            ] {
                let mut selector = Selector::new(k);
                for value in &input {
                    selector.offer(*value, &mut |a, b| Ok(a.cmp(b))).unwrap();
                }
                selector.finish(&mut |a, b| Ok(a.cmp(b))).unwrap();
                let mut expected = input;
                expected.sort();
                expected.truncate(k);
                assert_eq!(selector.buffer, expected);
            }
        }
    }

    #[test]
    fn merge_reinserts_before_the_last_requested_output() {
        let mut seen = vec![];
        let result = merge_runs(2, 4, 1, &mut |left, right| {
            seen.push((left, right));
            if left == 1 {
                return Err(failure("advanced left"));
            }
            // left run indices 0,1 have values 0,2; right indices 2,3 values 1,3.
            Ok([0, 2, 1, 3][left].cmp(&[0, 2, 1, 3][right]))
        });
        assert!(result.unwrap_err().to_string().contains("advanced left"));
        assert_eq!(seen, [(2, 0), (1, 2)]);
    }

    #[test]
    fn merge_queue_ties_prefer_the_other_run_after_reinsertion() {
        let mut seen = vec![];
        let result = merge_runs(2, 4, 4, &mut |left, right| {
            seen.push((left, right));
            Ok(Ordering::Equal)
        })
        .unwrap();
        assert_eq!(result, [0, 2, 1, 3]);
        assert_eq!(seen, [(2, 0), (1, 2), (3, 1)]);
        assert_eq!(
            merge_runs(0, 2, 2, &mut |_, _| Err(failure("no compare"))).unwrap(),
            [0, 1]
        );
        assert_eq!(
            merge_runs(2, 2, 2, &mut |_, _| Err(failure("no compare"))).unwrap(),
            [0, 1]
        );
    }

    #[test]
    fn mini_sort_has_java_run_detection_comparison_order() {
        let mut values = vec![4, 2, 3, 1];
        let mut seen = vec![];
        java_small_sort(&mut values, &mut |a, b| {
            seen.push((*a, *b));
            Ok(a.cmp(b))
        })
        .unwrap();
        assert_eq!(values, [1, 2, 3, 4]);
        assert_eq!(seen, [(2, 4), (3, 2), (3, 4), (3, 2), (1, 3), (1, 2)]);
    }
}
