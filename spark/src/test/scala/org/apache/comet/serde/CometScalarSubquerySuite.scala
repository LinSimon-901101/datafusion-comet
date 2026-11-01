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

package org.apache.comet.serde

import java.util.concurrent.atomic.AtomicInteger

import org.apache.spark.sql.CometTestBase
import org.apache.spark.sql.catalyst.InternalRow
import org.apache.spark.sql.catalyst.expressions.{Alias, GenericInternalRow, GetStructField, Literal, NamedExpression}
import org.apache.spark.sql.comet.{CometExec, CometExecUtils, CometScalarSubquery => JVMScalarSubquery}
import org.apache.spark.sql.execution.{ProjectExec, ScalarSubquery, SubqueryExec}
import org.apache.spark.sql.types._
import org.apache.spark.unsafe.types.UTF8String

import org.apache.comet.CometSparkSessionExtensions.{isSpark40Plus, isSpark41Plus}
import org.apache.comet.serde.QueryPlanSerde.supportedDataType
import org.apache.comet.shims.CometTypeShim

/** Type-gate and native transfer tests that avoid optimizer folding. */
class CometScalarSubquerySuite extends CometTestBase with CometTypeShim {

  private def struct(dt: DataType): StructType = StructType(Seq(StructField("value", dt)))

  private lazy val emptyInput = spark.range(0).queryExecution.sparkPlan

  private def subquery(dt: DataType): ScalarSubquery = {
    // Inspect the declared result type without executing or optimizing the typed NULL away.
    val plan = ProjectExec(Seq(Alias(Literal.create(null, dt), "result")()), emptyInput)
    ScalarSubquery(SubqueryExec("type-check", plan), NamedExpression.newExprId)
  }

  private def supported(dt: DataType): Boolean =
    CometScalarSubquery.getSupportLevel(subquery(dt)) == Compatible()

  private def versionSpecificTypes: Seq[DataType] = {
    val strings =
      if (isSpark40Plus) Seq(DataType.fromDDL("STRING COLLATE UTF8_LCASE")) else Seq.empty
    val times = if (isSpark41Plus) Seq(DataType.fromDDL("TIME")) else Seq.empty
    strings ++ times ++ variantType.toSeq
  }

  private val scalarTypes: Seq[DataType] = Seq(
    BooleanType,
    ByteType,
    ShortType,
    IntegerType,
    LongType,
    FloatType,
    DoubleType,
    StringType,
    BinaryType,
    DateType,
    TimestampType,
    TimestampNTZType,
    NullType,
    DecimalType(1, 0),
    DecimalType(10, 2),
    DecimalType(38, 38))

  test("reused struct fields serialize once per native execution context") {
    val resultType =
      StructType(Seq(StructField("text", StringType), StructField("number", IntegerType)))
    val first = subquery(resultType)
    val other = subquery(resultType)
    val metadataType = StructType(
      resultType.fields.updated(
        0,
        resultType
          .fields(0)
          .copy(metadata = new MetadataBuilder().putLong("parquet.field.id", 41).build())))
    val metadataFirst = subquery(metadataType).copy(exprId = first.exprId)
    val range = OperatorOuterClass.Operator
      .newBuilder()
      .setRangeScan(
        OperatorOuterClass.RangeScan
          .newBuilder()
          .setStart(0)
          .setStep(1)
          .setNumElements(4)
          .setNumSlices(1))
      .build()
    val fields = Seq(
      Alias(GetStructField(first, 0), "text")(),
      Alias(GetStructField(first, 1), "number")(),
      Alias(GetStructField(other, 0), "other")())
    val projection = CometExec.serializeNativePlan(
      CometExecUtils.getProjectionNativePlan(fields, Seq.empty, range).get)
    val metadataProjection = CometExec.serializeNativePlan(
      CometExecUtils
        .getProjectionNativePlan(
          fields :+ Alias(GetStructField(metadataFirst, 0), "metadataText")(),
          Seq.empty,
          range)
        .get)

    val counts = spark.sparkContext
      .parallelize(Seq(0), 1)
      .mapPartitions { _ =>
        (0 until 3).map { generation =>
          val firstReads = new AtomicInteger()
          val otherReads = new AtomicInteger()
          val text = s"$generation-" + ("wide" * 16384)
          def row(value: String, reads: AtomicInteger): InternalRow =
            new GenericInternalRow(Array[Any](UTF8String.fromString(value), generation)) {
              // The IPC serializer reads this field once; isNull and scalar evaluation do not.
              override def getUTF8String(ordinal: Int): UTF8String = {
                reads.incrementAndGet()
                super.getUTF8String(ordinal)
              }
            }
          val firstValue = if (generation == 2) null else row(text, firstReads)
          val otherValue = row(s"other-$generation", otherReads)
          val firstResult = new ScalarSubquery(first.plan, first.exprId) {
            override def eval(input: InternalRow): Any = firstValue
          }
          val otherResult = new ScalarSubquery(other.plan, other.exprId) {
            override def eval(input: InternalRow): Any = otherValue
          }
          val (plan, columns) =
            if (generation == 1) (metadataProjection, 4) else (projection, 3)
          val iterator = CometExec.getCometIterator(Array.empty[Object], columns, plan, 1, 0)
          try {
            JVMScalarSubquery.setSubquery(iterator.id, firstResult)
            JVMScalarSubquery.setSubquery(iterator.id, otherResult)
            var rows = 0
            while (iterator.hasNext) {
              val batch = iterator.next()
              (0 until batch.numRows()).foreach { index =>
                if (generation == 2) {
                  assert(batch.column(0).isNullAt(index))
                  assert(batch.column(1).isNullAt(index))
                } else {
                  assert(batch.column(0).getUTF8String(index).toString == text)
                  assert(batch.column(1).getInt(index) == generation)
                }
                assert(batch.column(2).getUTF8String(index).toString == s"other-$generation")
                if (generation == 1) {
                  assert(batch.column(3).getUTF8String(index).toString == text)
                }
                rows += 1
              }
            }
            assert(rows == 4)
            (firstReads.get(), otherReads.get())
          } finally {
            JVMScalarSubquery.removeSubquery(iterator.id, firstResult)
            JVMScalarSubquery.removeSubquery(iterator.id, otherResult)
            iterator.close()
          }
        }.iterator
      }
      .collect()
      .toSeq
    assert(counts == Seq((1, 1), (2, 1), (0, 1)))
  }

  test("non-struct scalar subqueries use the shared type gate defaults") {
    val types = scalarTypes ++ versionSpecificTypes ++ Seq(
      ArrayType(IntegerType),
      MapType(StringType, IntegerType),
      CalendarIntervalType,
      YearMonthIntervalType(),
      DayTimeIntervalType(),
      CharType(5),
      VarcharType(5))
    types.foreach(dt => assert(supported(dt) == supportedDataType(dt), dt))
  }

  test("struct scalar subqueries retain supported fields and case-distinct names") {
    val distinct =
      StructType(Seq(StructField("a", IntegerType), StructField("A", LongType)))
    (scalarTypes.map(struct) ++ scalarTypes.map(dt => struct(struct(dt))) ++
      Seq(distinct, struct(distinct))).foreach { dt =>
      assert(supported(dt), dt)
      assert(CometScalarSubquery.convert(subquery(dt), Seq.empty, binding = false).isDefined, dt)
    }
  }

  test("struct scalar subqueries reject unsupported fields and shapes recursively") {
    val rejected = versionSpecificTypes ++ Seq(
      ArrayType(IntegerType),
      MapType(StringType, IntegerType),
      StructType(Nil),
      StructType(Seq(StructField("same", IntegerType), StructField("same", LongType))),
      CalendarIntervalType,
      YearMonthIntervalType(),
      DayTimeIntervalType(),
      CharType(5),
      VarcharType(5))
    rejected.foreach { dt =>
      Seq(struct(dt), struct(struct(dt))).foreach { nested =>
        assert(!supported(nested), nested)
      }
    }
    rejected.collect { case s: StructType => s }.foreach(s => assert(!supported(s), s))
  }

  test("struct decimal scale restrictions do not change the legacy non-struct gate") {
    withSQLConf("spark.sql.legacy.allowNegativeScaleOfDecimal" -> "true") {
      val negative = DecimalType(10, -2)
      assert(supportedDataType(struct(negative), allowComplex = true))
      assert(supported(negative))
      assert(!supported(struct(negative)))
      assert(!supported(struct(struct(negative))))
      Seq(DecimalType(1, 0), DecimalType(38, 0), DecimalType(38, 38)).foreach { dt =>
        assert(supported(struct(dt)), dt)
        assert(supported(struct(struct(dt))), dt)
      }
    }
  }
}
