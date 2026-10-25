/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package org.apache.spark.sql.comet

import scala.jdk.CollectionConverters._

import org.apache.spark.TaskContext
import org.apache.spark.rdd.{ParallelCollectionRDD, RDD}
import org.apache.spark.serializer.Serializer
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Attribute, AttributeSet, BoundReference, GetStructField, Literal, NamedExpression, SortOrder, UnsafeProjection}
import org.apache.spark.sql.catalyst.util.truncatedString
import org.apache.spark.sql.comet.execution.arrow.{CometArrowConverters, CometArrowStream}
import org.apache.spark.sql.comet.execution.shuffle.{CometShuffledBatchRDD, CometShuffleExchangeExec}
import org.apache.spark.sql.comet.util.{Utils => CometUtils}
import org.apache.spark.sql.execution.{ScalarSubquery, SparkPlan, TakeOrderedAndProjectExec, UnaryExecNode, UnsafeRowSerializer}
import org.apache.spark.sql.execution.metric.{SQLMetric, SQLMetrics, SQLShuffleReadMetricsReporter, SQLShuffleWriteMetricsReporter}
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.vectorized.ColumnarBatch

import org.apache.comet.{CometConf, ConfigEntry, Native, SparkErrorConverter}
import org.apache.comet.CometSparkSessionExtensions.isCometShuffleEnabled
import org.apache.comet.exceptions.CometQueryExecutionException
import org.apache.comet.serde.{CometScalarSubquery, Compatible, OperatorOuterClass, SupportLevel, Unsupported}
import org.apache.comet.serde.QueryPlanSerde.{exprToProto, operatorExprToProto, supportedSortType, withSharedSubexpressions}
import org.apache.comet.serde.operator.CometSink

object CometTakeOrderedAndProjectExec extends CometSink[TakeOrderedAndProjectExec] {

  override def enabledConfig: Option[ConfigEntry[Boolean]] = Some(
    CometConf.COMET_EXEC_TAKE_ORDERED_AND_PROJECT_ENABLED)

  override def getSupportLevel(op: TakeOrderedAndProjectExec): SupportLevel = {
    if (!isCometShuffleEnabled(op.conf)) {
      return Unsupported(Some("TakeOrderedAndProject requires shuffle to be enabled"))
    }
    val projectProtos = op.projectList.map { p =>
      val o = exprToProto(p, op.child.output)
      if (o.isEmpty) {
        return Unsupported(Some(s"unsupported projection: $p"))
      }
      o.get
    }
    org.apache.comet.serde.CometScalarSubquery
      .projectionConsumerFallbackReason(op.projectList, projectProtos)
      .foreach { reason =>
        return Unsupported(Some(reason))
      }
    CometScalarSubquery.withDeferredTopK(
      CometScalarSubquery.usesDeferredTopK(op.sortOrder, op.limit, op.projectList)) {
      op.sortOrder.foreach { s =>
        val o = operatorExprToProto(s, op.child.output)
        if (o.isEmpty) {
          return Unsupported(Some(s"unsupported sort order: $s"))
        }
      }
    }
    if (!supportedSortType(op, op.sortOrder)) {
      return Unsupported(Some("unsupported data type in sort order"))
    }
    Compatible()
  }

  override def createExec(
      nativeOp: OperatorOuterClass.Operator,
      op: TakeOrderedAndProjectExec): CometNativeExec = {
    CometSinkPlaceHolder(
      nativeOp,
      op,
      CometTakeOrderedAndProjectExec(
        op,
        op.output,
        op.limit,
        op.offset,
        op.sortOrder,
        op.projectList,
        CometLocalTopKExec.create(op).getOrElse(op.child)))
  }
}

/**
 * Comet physical plan node for Spark `TakeOrderedAndProjectExec`.
 *
 * It is used to execute a `TakeOrderedAndProjectExec` physical operator by using Comet native
 * engine. It is not like other physical plan nodes which are wrapped by `CometExec`, because it
 * contains two native executions separated by a Comet shuffle exchange.
 */
case class CometTakeOrderedAndProjectExec(
    override val originalPlan: SparkPlan,
    override val output: Seq[Attribute],
    limit: Int,
    offset: Int,
    sortOrder: Seq[SortOrder],
    projectList: Seq[NamedExpression],
    child: SparkPlan)
    extends CometExec
    with UnaryExecNode {

  override def producedAttributes: AttributeSet = outputSet ++ AttributeSet(projectList)

  private lazy val writeMetrics =
    SQLShuffleWriteMetricsReporter.createShuffleWriteMetrics(sparkContext)

  private lazy val readMetrics =
    SQLShuffleReadMetricsReporter.createShuffleReadMetrics(sparkContext)

  override lazy val metrics: Map[String, SQLMetric] = Map(
    "dataSize" -> SQLMetrics.createSizeMetric(sparkContext, "data size"),
    "numPartitions" -> SQLMetrics.createMetric(
      sparkContext,
      "number of partitions")) ++ readMetrics ++ writeMetrics ++ CometMetricNode.shuffleMetrics(
    sparkContext)

  private lazy val serializer: Serializer =
    new UnsafeRowSerializer(child.output.size, longMetric("dataSize"))

  // Exposed for testing.
  lazy val orderingSatisfies: Boolean =
    SortOrder.orderingSatisfies(child.outputOrdering, sortOrder)

  lazy val deferSortErrors =
    CometScalarSubquery.usesDeferredTopK(sortOrder, limit, projectList)

  // Also exposed to tests so they can inspect the plan actually used for final execution.
  def finalNativePlan(inputPartitions: Int): Option[OperatorOuterClass.Operator] = {
    val selectedLocally = child match {
      case local: CometLocalTopKExec =>
        inputPartitions == 1 && local.limit == limit && local.sortOrder == sortOrder
      case _ => false
    }
    val selection = if (selectedLocally) {
      // The local TopK has already ordered and bounded the only input partition. Spark's
      // physical limit includes the offset; the native limit applies that offset once.
      CometExecUtils.getLimitNativePlan(child.output, limit, offset)
    } else {
      topKNativePlan(offset)
    }
    // Spark's projection runs on its own over the rows the sort keeps, so only its list can share
    // a subexpression (see QueryPlanSerde.canShortCircuitNulls)
    withSharedSubexpressions(projectList, eagerSharing = true) {
      selection.flatMap(CometExecUtils.getProjectionNativePlan(projectList, child.output, _))
    }
  }

  // Spark's ordering evaluates each sort key on its own, without subexpression elimination
  private def topKNativePlan(offset: Int): Option[OperatorOuterClass.Operator] =
    withSharedSubexpressions(Seq.empty) {
      CometExecUtils.getTopKNativePlan(
        child.output,
        sortOrder,
        child,
        limit,
        offset,
        deferSortErrors)
    }

  override def executeCollect(): Array[InternalRow] = {
    if (!deferSortErrors) {
      super.executeCollect()
    } else {
      // Spark's executeQuery prepares and awaits scalar subqueries before entering this body.
      executeQuery {
        val childRDD = child.executeColumnar()
        if (childRDD.getNumPartitions == 0 || limit == 0) {
          Array.empty[InternalRow]
        } else {
          val inputAttributes = child.output
          val fetch = limit

          // Resolve cached subquery values solely for the standalone driver merge. Keep the
          // original expressions for task execution and its normal subquery registration.
          // Fold only struct-field extraction: evaluating Add here could expose a skipped error.
          val mergeOrders = sortOrder.map { order =>
            order
              .transformUp {
                case bound: BoundReference => inputAttributes(bound.ordinal)
                case subquery: ScalarSubquery =>
                  Literal.create(subquery.eval(InternalRow.empty), subquery.dataType)
                case field: GetStructField if field.child.isInstanceOf[Literal] =>
                  Literal.create(field.eval(InternalRow.empty), field.dataType)
              }
              .asInstanceOf[SortOrder]
          }
          val referenced = AttributeSet(mergeOrders.flatMap(_.child.references))
          val mergeAttributes = inputAttributes.filter(referenced.contains)
          val mergeSchema = CometUtils.fromAttributes(mergeAttributes)
          val mergeSort = OperatorOuterClass.Sort
            .newBuilder()
            .setFetch(fetch)
            .setSkip(0)
            .setDeferSortErrors(true)
          withSharedSubexpressions(Seq.empty) {
            CometScalarSubquery.withDeferredTopK(true) {
              mergeOrders.foreach { order =>
                mergeSort.addSortOrders(operatorExprToProto(order, mergeAttributes).get)
              }
            }
          }
          // Fresh expression serialization keeps query contexts inline; do not intern a detached
          // Sort, since it has no Operator.sql_text_pool for the standalone planner to resolve.
          val mergeSortBytes = mergeSort.build().toByteArray

          // Match RDD.takeOrdered: partition zero contributes an empty array when empty; other
          // empty partitions contribute nothing. RDD.reduce merges results in completion order.
          val localArrays = deferredCollectLocalTopK(childRDD)
          val mergeConf = conf.clone()
          val limited = localArrays.reduce { (left, right) =>
            SQLConf.withExistingConf(mergeConf) {
              if (left.isEmpty) {
                right
              } else if (right.isEmpty) {
                left
              } else {
                val combined = left ++ right
                val selectInputs = UnsafeProjection.create(mergeAttributes, inputAttributes)
                selectInputs.initialize(0)
                val inputs = combined.iterator.map(row => selectInputs(row).copy()).toSeq
                val ipc = CometArrowConverters.serializeRows(inputs, mergeSchema)
                val selected =
                  try {
                    new Native().mergeTopK(ipc, mergeSortBytes, left.length, fetch)
                  } catch {
                    case error: CometQueryExecutionException =>
                      throw SparkErrorConverter.convertToSparkException(error)
                  }
                selected.map(combined(_))
              }
            }
          }
          val data = if (offset > 0) limited.drop(offset) else limited
          if (projectList != inputAttributes) {
            // Admission restricts this path to direct attributes, so this only selects/reorders
            // already-owned payload values and cannot evaluate a fallible scalar expression.
            val project = UnsafeProjection.create(projectList, inputAttributes)
            project.initialize(0)
            data.map(row => project(row).copy(): InternalRow)
          } else {
            data
          }
        }
      }
    }
  }

  private def deferredCollectLocalTopK(childRDD: RDD[ColumnarBatch]): RDD[Array[InternalRow]] = {
    // Spark collect applies takeOrdered even after its ordering-satisfies limit shortcut.
    // Unlike doExecuteColumnar, this path must still run the native selector on that prefix.
    val input = if (orderingSatisfies) {
      CometExecUtils.getNativeLimitRDD(childRDD, child.output, limit)
    } else {
      childRDD
    }
    val numParts = input.getNumPartitions
    val inputAttributes = child.output
    val inputSchema = CometUtils.fromAttributes(inputAttributes)
    val inputTypes = inputAttributes.map(_.dataType)
    val serializedTopK = CometExec.serializeNativePlan(topKNativePlan(offset = 0).get)
    input.mapPartitionsWithIndexInternal { case (idx, iter) =>
      val native = CometExec.getCometIterator(
        CometArrowStream.inputObjects(iter, inputSchema, "CometTakeOrderedAndProject-collect"),
        inputAttributes.length,
        serializedTopK,
        numParts,
        idx,
        inputTypes)
      setSubqueries(native.id, this)
      Option(TaskContext.get()).foreach { context =>
        context.addTaskCompletionListener[Unit] { _ =>
          cleanSubqueries(native.id, this)
        }
      }
      val copyProjection = UnsafeProjection.create(inputAttributes, inputAttributes)
      copyProjection.initialize(idx)
      val rows: Array[InternalRow] = native.flatMap { batch =>
        batch.rowIterator().asScala.map[InternalRow](row => copyProjection(row).copy())
      }.toArray
      if (rows.nonEmpty || idx == 0) Iterator.single(rows) else Iterator.empty
    }
  }

  protected override def doExecuteColumnar(): RDD[ColumnarBatch] = {
    val childRDD = child.executeColumnar()
    if (childRDD.getNumPartitions == 0 || limit == 0) {
      new ParallelCollectionRDD(sparkContext, Seq.empty[ColumnarBatch], 1, Map.empty)
    } else {
      val singlePartitionRDD = if (childRDD.getNumPartitions == 1) {
        childRDD
      } else {
        val localTopK = if (child.isInstanceOf[CometLocalTopKExec]) {
          // A fused child is already ordered and bounded. Check it before orderingSatisfies,
          // which is also true for that child, to avoid adding a redundant native limit.
          childRDD
        } else if (orderingSatisfies) {
          CometExecUtils.getNativeLimitRDD(childRDD, child.output, limit)
        } else {
          val numParts = childRDD.getNumPartitions
          // Serialize the plan once before mapping to avoid repeated serialization per partition
          val topK = topKNativePlan(offset = 0).get
          val serializedTopK = CometExec.serializeNativePlan(topK)
          val numOutputCols = child.output.length
          val inputSchema = CometUtils.fromAttributes(child.output)
          childRDD.mapPartitionsWithIndexInternal { case (idx, iter) =>
            val it = CometExec.getCometIterator(
              CometArrowStream.inputObjects(iter, inputSchema, "CometTakeOrderedAndProject-topK"),
              numOutputCols,
              serializedTopK,
              numParts,
              idx,
              child.output.map(_.dataType))
            setSubqueries(it.id, this)

            Option(TaskContext.get()).foreach { context =>
              context.addTaskCompletionListener[Unit] { _ =>
                cleanSubqueries(it.id, this)
              }
            }

            it
          }
        }

        // Shuffle to Single Partition using Comet native shuffle
        val dep = CometShuffleExchangeExec.prepareShuffleDependency(
          localTopK,
          child.output,
          outputPartitioning,
          serializer,
          metrics)
        metrics("numPartitions").set(dep.partitioner.numPartitions.toLong)

        new CometShuffledBatchRDD(dep, readMetrics)
      }

      // Serialize the plan once before mapping to avoid repeated serialization per partition
      val serializedFinalPlan =
        CometExec.serializeNativePlan(finalNativePlan(childRDD.getNumPartitions).get)
      val finalOutputTypes = output.map(_.dataType)
      val finalInputSchema = CometUtils.fromAttributes(child.output)
      singlePartitionRDD.mapPartitionsInternal { iter =>
        val it = CometExec.getCometIterator(
          CometArrowStream
            .inputObjects(iter, finalInputSchema, "CometTakeOrderedAndProject-final"),
          finalOutputTypes.length,
          serializedFinalPlan,
          1,
          0,
          finalOutputTypes)
        setSubqueries(it.id, this)

        Option(TaskContext.get()).foreach { context =>
          context.addTaskCompletionListener[Unit] { _ =>
            cleanSubqueries(it.id, this)
          }
        }

        it
      }
    }
  }

  override def simpleString(maxFields: Int): String = {
    val orderByString = truncatedString(sortOrder, "[", ",", "]", maxFields)
    val outputString = truncatedString(output, "[", ",", "]", maxFields)

    s"CometTakeOrderedAndProjectExec(limit=$limit, offset=$offset, " +
      s"orderBy=$orderByString, output=$outputString)"
  }

  override protected def withNewChildInternal(newChild: SparkPlan): SparkPlan =
    this.copy(child = newChild)
}
