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

import scala.jdk.CollectionConverters._
import scala.util.DynamicVariable

import org.apache.spark.sql.catalyst.expressions.{Add, Alias, And, ArrayContains, ArrayPosition, ArrayRemove, ArraysOverlap, ArraysZip, ArrayUnion, Ascii, Attribute, BloomFilterMightContain, BoundReference, CaseWhen, Cast, Coalesce, Concat, ConcatWs, CreateArray, CreateMap, CreateNamedStruct, DateFormatClass, DateFromUnixDate, DayOfWeek, Divide, DivModLike, ElementAt, EqualNullSafe, EqualTo, EvalMode, Expression, ExprId, GetArrayItem, GetArrayStructFields, GetMapValue, GetStructField, GreaterThan, GreaterThanOrEqual, Greatest, Hour, If, In, InSet, IntegralDivide, IsNaN, IsNotNull, IsNull, KnownFloatingPointNormalized, KnownNotNull, KnownNullable, LambdaFunction, Least, Length, LessThan, LessThanOrEqual, Literal, Log, Log10, Log2, Lower, MapFromArrays, Minute, Multiply, Murmur3Hash, NamedExpression, Not, OctetLength, Or, Pmod, PreciseTimestampConversion, Rand, RegExpExtract, RegExpExtractAll, Remainder, Reverse, Round, ScalaUDF, Second, Size, Slice, SortOrder, SpecifiedWindowFrame, StringLPad, StringRepeat, StringRPad, Subtract, ToUnixTimestamp, TryEval, UnixDate, UnixTimestamp, Upper, WeekDay, WindowSpecDefinition, XxHash64}
import org.apache.spark.sql.catalyst.expressions.aggregate.Sum
import org.apache.spark.sql.catalyst.expressions.objects.StaticInvoke
import org.apache.spark.sql.catalyst.optimizer.NormalizeNaNAndZero
import org.apache.spark.sql.catalyst.trees.TreeNodeTag
import org.apache.spark.sql.catalyst.trees.TreePattern.PLAN_EXPRESSION
import org.apache.spark.sql.comet.CometExec
import org.apache.spark.sql.execution.{BaseSubqueryExec, ExecSubqueryExpression, ExpandExec, GenerateExec, ProjectExec, ReusedSubqueryExec, ScalarSubquery, SparkPlan, TakeOrderedAndProjectExec, UnionExec}
import org.apache.spark.sql.execution.adaptive.{AdaptiveSparkPlanExec, QueryStageExec}
import org.apache.spark.sql.execution.aggregate.BaseAggregateExec
import org.apache.spark.sql.execution.exchange.ReusedExchangeExec
import org.apache.spark.sql.internal.SQLConf
import org.apache.spark.sql.types._

import com.google.protobuf.Message

import org.apache.comet.CometConf
import org.apache.comet.CometSparkSessionExtensions.isSpark40Plus
import org.apache.comet.CometSparkSessionExtensions.withFallbackReason
import org.apache.comet.serde.QueryPlanSerde.{serializeDataType, supportedDataType}
import org.apache.comet.shims.{CometEvalModeUtil, CometExprTraitShim}

object CometScalarSubquery extends CometExpressionSerde[ScalarSubquery] with CometExprTraitShim {

  private val structSubqueryInputs =
    new TreeNodeTag[Set[ExprId]]("CometStructScalarSubqueryInputs")
  private val structSubqueryInputBounds =
    new TreeNodeTag[Map[ExprId, Long]]("CometStructScalarSubqueryInputBounds")
  private val structSubqueryRowPath =
    new TreeNodeTag[Boolean]("CometStructScalarSubqueryRowPath")
  private val currentStructSubqueryInputBounds = new DynamicVariable[Map[ExprId, Long]](Map.empty)
  private val structSubqueryRowPathReason =
    "Materialized variable-width struct scalar subquery aliases require Spark rows"

  private[comet] def rowPathFallbackReason(plan: SparkPlan): Option[String] = {
    if (!plan.getTagValue(structSubqueryRowPath).contains(true)) return None
    val original = plan match {
      case comet: CometExec => comet.originalPlan
      case other => other
    }
    // Preserve a more specific expression diagnostic before the row-path safety gate.
    val projected = original match {
      case project: ProjectExec => project.projectList
      case topK: TakeOrderedAndProjectExec => topK.projectList
      case _ => Seq.empty
    }
    val reason = withPlanInputs(plan) {
      projected.iterator.map(consumerFallbackReason).collectFirst { case Some(reason) =>
        reason
      }
    }
    reason
      .orElse {
        projected.iterator
          .flatMap(_.collect { case e: Expression => e })
          .flatMap { expr =>
            QueryPlanSerde.exprSerdeMap.get(expr.getClass).flatMap { handler =>
              handler.asInstanceOf[CometExpressionSerde[Expression]].getSupportLevel(expr) match {
                case Unsupported(notes) => notes
                case _ => None
              }
            }
          }
          .take(1)
          .toSeq
          .headOption
      }
      .orElse(Some(structSubqueryRowPathReason))
  }

  private def provenanceChildren(plan: SparkPlan): Seq[SparkPlan] = plan match {
    case reused: ReusedExchangeExec => Seq(reused.child)
    case stage: QueryStageExec => Seq(stage.plan)
    case adaptive: AdaptiveSparkPlanExec => Seq(adaptive.executedPlan)
    case comet: CometExec if comet.children.isEmpty => Seq(comet.originalPlan)
    case other => other.children
  }

  private val structSubqueryContext =
    new TreeNodeTag[Boolean]("CometStructScalarSubqueryContext")
  private val currentStructSubqueryInputs = new DynamicVariable[Set[ExprId]](Set.empty)
  private val currentPlanHasStructSubquery = new DynamicVariable[Boolean](false)
  private val deferredTopKSort = new DynamicVariable[Boolean](false)

  private def topKValue(expr: Expression): Boolean = expr match {
    case _: Attribute | _: BoundReference | _: Literal | _: ScalarSubquery =>
      Set[DataType](IntegerType, LongType).contains(expr.dataType)
    case field: GetStructField =>
      def scalarField(e: Expression): Boolean = e match {
        case _: ScalarSubquery => true
        case f: GetStructField => scalarField(f.child)
        case _ => false
      }
      Set[DataType](IntegerType, LongType).contains(field.dataType) && scalarField(field)
    case cast: Cast if Cast.canUpCast(cast.child.dataType, cast.dataType) =>
      Set[DataType](IntegerType, LongType).contains(cast.dataType) && topKValue(cast.child)
    case _ => false
  }

  private def deferredTopKKey(expr: Expression): Boolean = expr match {
    case sort: SortOrder => deferredTopKKey(sort.child)
    case add: Add if add.evalMode == EvalMode.ANSI =>
      Set[DataType](IntegerType, LongType).contains(add.dataType) &&
      add.children.forall(topKValue)
    case _ => topKValue(expr)
  }

  // A buffer smaller than Java TimSort's MIN_MERGE uses its fixed insertion-sort path.
  // The native selector mirrors that path and Guava's comparisons, including skipped keys.
  def usesDeferredTopK(
      sortOrder: Seq[SortOrder],
      limit: Int,
      projectList: Seq[NamedExpression]): Boolean =
    isSpark40Plus && !java.lang.Boolean.getBoolean("java.util.Arrays.useLegacyMergeSort") &&
      limit > 0 && limit < 16 && projectList.forall(_.isInstanceOf[Attribute]) &&
      sortOrder.forall(deferredTopKKey) &&
      sortOrder.exists(_.exists {
        case add: Add => add.evalMode == EvalMode.ANSI
        case _ => false
      }) && sortOrder.exists(_.exists(_.isInstanceOf[ScalarSubquery]))

  def withDeferredTopK[T](enabled: Boolean)(f: => T): T =
    deferredTopKSort.withValue(enabled)(f)

  /**
   * Trace struct scalar values before native conversion replaces their producing projections.
   * Attribute object identity is insufficient: aliases, nullable join outputs and reused stages
   * can create fresh attributes. Keep provenance by ExprId on the owning plan instead.
   *
   * This is recomputed on every rule application, including the plans hidden by AQE/reuse nodes.
   * No provenance is inferred from a column's data type or carried in a global ExprId registry.
   */
  private[comet] def tagPlanInputs(root: SparkPlan): Unit = {
    val visited = new java.util.IdentityHashMap[SparkPlan, Set[ExprId]]()
    def remember(plan: SparkPlan, sources: Set[ExprId]): Unit = {
      val _ = visited.put(plan, sources)
    }

    def visit(plan: SparkPlan): Set[ExprId] = {
      if (visited.containsKey(plan)) return visited.get(plan)
      remember(plan, Set.empty)
      plan.unsetTagValue(structSubqueryInputs)
      plan.unsetTagValue(structSubqueryContext)
      plan.unsetTagValue(structSubqueryInputBounds)
      plan.unsetTagValue(structSubqueryRowPath)

      val original = plan match {
        case comet: CometExec => comet.originalPlan
        case other => other
      }
      val children = provenanceChildren(plan)
      val childSources = children.map(visit)
      var sources = childSources.flatten.toSet
      var bounds =
        children.flatMap(_.getTagValue(structSubqueryInputBounds).getOrElse(Map.empty)).toMap
      def boundFor(expression: Expression): Option[Long] =
        currentStructSubqueryInputBounds.withValue(bounds)(boundedResultBytes(expression))
      def rememberBound(id: ExprId, bound: Option[Long]): Unit = bound match {
        case Some(bytes) => bounds += id -> bytes
        case None => bounds -= id
      }
      // A struct subquery can enable native conversion of both its operator and ancestors.
      // Keep that context even when a later projection drops the struct: sibling/ancestor
      // calendar consumers can still receive ordinary columns outside the native range.
      val hasStructContext =
        children.exists(_.getTagValue(structSubqueryContext).contains(true)) ||
          original.expressions.exists(hasStructSubquerySource(_, sources))
      plan.setTagValue(structSubqueryContext, hasStructContext)
      if (sources.isEmpty &&
        !original.expressions.exists(_.containsPattern(PLAN_EXPRESSION))) {
        return Set.empty
      }

      // Union and reused stages can expose different ExprIds for the same output positions.
      original match {
        case _: UnionExec | _: ReusedExchangeExec | _: QueryStageExec |
            _: AdaptiveSparkPlanExec =>
          plan.output.zipWithIndex.foreach { case (output, ordinal) =>
            if (children.zip(childSources).exists { case (child, ids) =>
                child.output.lift(ordinal).exists(a => ids.contains(a.exprId))
              }) {
              sources += output.exprId
              val positionBounds = children.map { child =>
                child.output.lift(ordinal).flatMap { attribute =>
                  val known = child.getTagValue(structSubqueryInputBounds).getOrElse(Map.empty)
                  currentStructSubqueryInputBounds.withValue(known)(boundedResultBytes(attribute))
                }
              }
              rememberBound(
                output.exprId,
                if (positionBounds.forall(_.isDefined)) Some(positionBounds.flatten.max)
                else None)
            }
          }
        case _ =>
      }

      val named = original.expressions.flatMap(_.collect { case e: NamedExpression => e })
      var changed = true
      while (changed) {
        val previous = sources
        val previousBounds = bounds
        named.foreach { expression =>
          if (hasStructSubquerySource(expression, sources)) {
            sources += expression.exprId
            rememberBound(expression.exprId, boundFor(expression))
          }
        }
        original match {
          case aggregate: BaseAggregateExec =>
            aggregate.aggregateExpressions.zip(aggregate.aggregateAttributes).foreach {
              case (expression, output) =>
                val function = expression.aggregateFunction
                if (hasStructSubquerySource(expression, sources) ||
                  function.inputAggBufferAttributes.exists(a => sources.contains(a.exprId))) {
                  sources += output.exprId
                  sources ++= function.aggBufferAttributes.map(_.exprId)
                  sources ++= function.inputAggBufferAttributes.map(_.exprId)
                }
            }
          case generate: GenerateExec if hasStructSubquerySource(generate.generator, sources) =>
            // Generators expose fresh attributes rather than NamedExpression aliases.
            sources ++= generate.generatorOutput.map(_.exprId)
          case expand: ExpandExec =>
            expand.output.zipWithIndex.foreach { case (output, ordinal) =>
              if (expand.projections.exists(p => hasStructSubquerySource(p(ordinal), sources))) {
                sources += output.exprId
              }
            }
          case _ =>
        }
        changed = previous != sources || previousBounds != bounds
      }

      plan.setTagValue(structSubqueryInputs, sources)
      plan.setTagValue(structSubqueryInputBounds, bounds)
      // ProjectionExec maps each input batch to one output batch, so a native producer's
      // byte cap survives field reads and projecting retained aliases. Other operators can
      // coalesce those batches again. Once they use Spark rows, keep wide consumers on rows
      // until a projection drops or reduces the wide alias before re-entering Arrow.
      def wideOutput(p: SparkPlan): Boolean = {
        val ids = p.getTagValue(structSubqueryInputs).getOrElse(Set.empty)
        val known = p.getTagValue(structSubqueryInputBounds).getOrElse(Map.empty)
        currentStructSubqueryInputBounds.withValue(known) {
          p.output.exists(a => ids.contains(a.exprId) && !fitsConsumerBatch(a))
        }
      }
      // Later expressions have columns rather than the native literal's provenance. An
      // inherited input cap protects reads, not new unbounded intermediates such as repeat.
      val materializedInputs = childSources.flatten.toSet
      def constructsUnboundedValue(expression: Expression): Boolean = {
        val selectsExistingValue = expression match {
          case _: Attribute | _: GetStructField | _: Alias | _: KnownNotNull | _: KnownNullable |
              _: TryEval | _: SortOrder | _: If | _: CaseWhen | _: Coalesce | _: Least |
              _: Greatest | _: WindowSpecDefinition | _: SpecifiedWindowFrame =>
            true
          case _ => false
        }
        (!selectsExistingValue && !fitsConsumerBatch(expression)) ||
        expression.children.exists(constructsUnboundedValue)
      }
      // A generator's ArrayType describes rows, not an allocated value. Its children are
      // the actual expression arguments, matching the GenerateExec serializer's boundary.
      val valueRoots = original match {
        case generate: GenerateExec => generate.generator.children
        case other => other.expressions
      }
      val expandsMaterializedInput = currentStructSubqueryInputBounds.withValue(bounds) {
        valueRoots.exists { expression =>
          // Retained input columns no longer carry the native literal's provenance. Gate
          // the root once, then inspect all branches, including independent constant work.
          expression.references.exists(a => materializedInputs.contains(a.exprId)) &&
          constructsUnboundedValue(expression)
        }
      }
      val wideChildren = children.filter(wideOutput)
      val opaqueWideProjection = original.isInstanceOf[ProjectExec] &&
        (wideOutput(plan) || wideChildren.nonEmpty) && withPlanInputs(plan) {
          QueryPlanSerde.withSharedSubexpressions(valueRoots, eagerSharing = true) {
            valueRoots.exists(e =>
              requiresRowProjection(e) || consumerFallbackReason(e).isDefined)
          }
        }
      if (expandsMaterializedInput || opaqueWideProjection || (wideChildren.nonEmpty &&
          (!original.isInstanceOf[ProjectExec] ||
            wideChildren.exists(_.getTagValue(structSubqueryRowPath).contains(true))))) {
        plan.setTagValue(structSubqueryRowPath, true)
        // Avoid copying wide aliases into Arrow immediately before their Spark row consumer.
        def keepWideInputsOnRows(p: SparkPlan): Unit = {
          if (wideOutput(p)) {
            p.setTagValue(structSubqueryRowPath, true)
            provenanceChildren(p).foreach(keepWideInputsOnRows)
          }
        }
        wideChildren.foreach(keepWideInputsOnRows)
      }
      val outputs = plan.output.iterator.map(_.exprId).filter(sources.contains).toSet
      remember(plan, outputs)
      outputs
    }

    val _ = visit(root)
  }

  /** Keep the lineage local to this conversion, even when an attribute was copied or rebound. */
  private[comet] def withPlanInputs[T](plan: SparkPlan)(f: => T): T = {
    val inputs = plan.getTagValue(structSubqueryInputs).getOrElse(Set.empty)
    val bounds = plan.getTagValue(structSubqueryInputBounds).getOrElse(Map.empty)
    currentStructSubqueryInputBounds.withValue(bounds) {
      currentStructSubqueryInputs.withValue(inputs) {
        currentPlanHasStructSubquery.withValue(
          plan.getTagValue(structSubqueryContext).getOrElse(false))(f)
      }
    }
  }

  private val calendarConsumerFallbackReason =
    "Calendar operations on struct scalar subqueries require Spark because their runtime values " +
      "may exceed the native calendar range"

  private val bloomInitializationFallbackReason =
    "Bloom initialization of struct scalar subqueries requires Spark evaluation"

  private val timezoneConsumerFallbackReason =
    "Timezone-dependent scalar subquery operations require Spark for non-fixed timezones"

  private val dateUnixTimestampFallbackReason =
    "Date scalar subqueries in unix_timestamp require Spark because their runtime values " +
      "may exceed the native calendar range"

  private def containsCalendarType(dt: DataType): Boolean = dt match {
    case DateType | TimestampType | TimestampNTZType => true
    case s: StructType => s.fields.exists(f => containsCalendarType(f.dataType))
    case ArrayType(elementType, _) => containsCalendarType(elementType)
    case MapType(keyType, valueType, _) =>
      containsCalendarType(keyType) || containsCalendarType(valueType)
    case _ => false
  }

  private val scalarInputBytes = 1024L * 1024

  private def sumBounds(bounds: Seq[Option[Long]], overhead: Long): Option[Long] =
    bounds.foldLeft(Option(overhead)) { (sum, next) =>
      for {
        a <- sum
        b <- next
        if a <= Long.MaxValue - b
      } yield a + b
    }

  private def fixedTypeBytes(dt: DataType): Option[Long] = dt match {
    case StringType | BinaryType | _: ArrayType | _: MapType => None
    case StructType(fields) => sumBounds(fields.toSeq.map(f => fixedTypeBytes(f.dataType)), 1)
    case _ => Some(dt.defaultSize.toLong + 1)
  }

  // This is an upper bound, never an average-width estimate for strings or array cardinality.
  private def boundedResultBytes(expr: Expression): Option[Long] = {
    val inputBound = expr match {
      case attribute: Attribute => currentStructSubqueryInputBounds.value.get(attribute.exprId)
      case _ => None
    }
    inputBound.orElse(fixedTypeBytes(expr.dataType)).orElse {
      expr match {
        case literal: Literal if literal.value == null => Some(0)
        case literal: Literal if literal.dataType == StringType =>
          Some(
            literal.value
              .asInstanceOf[org.apache.spark.unsafe.types.UTF8String]
              .numBytes()
              .toLong + 5)
        case literal: Literal if literal.dataType == BinaryType =>
          Some(literal.value.asInstanceOf[Array[Byte]].length.toLong + 5)
        case cast: Cast if cast.dataType == StringType =>
          val digits = cast.child.dataType match {
            case BooleanType => Some(5L)
            case decimal: DecimalType =>
              val precision = decimal.precision.toLong
              val scale = decimal.scale.toLong
              val plain =
                if (scale >= 0) math.max(precision + 2, scale + 3)
                else precision - scale + 1
              Some(math.max(plain, precision + 7))
            case ByteType => Some(4L)
            case ShortType => Some(6L)
            case IntegerType => Some(11L)
            case LongType => Some(20L)
            // Java-style float formatting uses at most 17 significant digits and a
            // signed three-digit exponent; leave room for sign, decimal point and E.
            case FloatType | DoubleType => Some(32L)
            case _ => None
          }
          digits.map(_ + 5)
        case padding: StringLPad => paddingBytes(padding.len)
        case padding: StringRPad => paddingBytes(padding.len)
        case repeat: StringRepeat =>
          repeat.right match {
            case Literal(count: Int, IntegerType) =>
              boundedResultBytes(repeat.left)
                .filter(_ <= Long.MaxValue / math.max(1, count))
                .map(_ * math.max(0, count))
            case _ => None
          }
        case concat: Concat => sumBounds(concat.children.map(boundedResultBytes), 5)
        case concat: ConcatWs if hasLiteralNullSeparator(concat) =>
          Some(0L)
        case array: CreateArray => sumBounds(array.children.map(boundedResultBytes), 5)
        case zip: ArraysZip if zip.children.forall(_.isInstanceOf[CreateArray]) =>
          val length = zip.children.map(_.children.length.toLong).foldLeft(0L)(math.max)
          // NULL padding still allocates a full fixed-width slot in every result field.
          sumBounds(
            zip.children.map(a => fixedTypeBytes(a.dataType.asInstanceOf[ArrayType].elementType)),
            1)
            .filter(_ <= (Long.MaxValue - 5) / math.max(1L, length))
            .map(_ * length + 5)
        case structure: CreateNamedStruct =>
          sumBounds(structure.valExprs.map(boundedResultBytes), 1)
        case conditional: If =>
          for {
            yes <- boundedResultBytes(conditional.trueValue)
            no <- boundedResultBytes(conditional.falseValue)
          } yield math.max(yes, no)
        case conditional: CaseWhen =>
          val values = conditional.branches.map(_._2) ++ conditional.elseValue.toSeq
          val bounds = values.map(boundedResultBytes)
          if (bounds.forall(_.isDefined)) Some(bounds.flatten.foldLeft(0L)(math.max)) else None
        case _: Coalesce | _: Greatest | _: Least =>
          val bounds = expr.children.map(boundedResultBytes)
          if (bounds.forall(_.isDefined)) Some(bounds.flatten.foldLeft(0L)(math.max)) else None
        case _: Alias | _: KnownNotNull | _: KnownNullable | _: TryEval | _: SortOrder =>
          expr.children.headOption.flatMap(boundedResultBytes)
        case _ => None
      }
    }
  }

  private def fitsConsumerBatch(expr: Expression): Boolean =
    boundedResultBytes(expr).exists(
      _ <= scalarInputBytes / math.max(1, CometConf.COMET_BATCH_SIZE.get()))

  private def hasLiteralNullSeparator(concat: ConcatWs): Boolean =
    concat.children.headOption match {
      case Some(Literal(null, _)) => true
      case _ => false
    }

  private def paddingBytes(length: Expression): Option[Long] = length match {
    case Literal(value: Int, IntegerType) => Some(math.max(0L, value.toLong) * 4 + 5)
    case Literal(null, _) => Some(0L)
    case _ => None
  }

  // Include intermediates such as repeat('x', numericSubquery), even when the source is integral.
  private def needsBoundedEvaluation(expr: Expression): Boolean =
    hasStructSubquery(expr) && expr.exists(e => !fitsConsumerBatch(e))

  private def borrowedArray(expr: Expression): Boolean = expr match {
    case _: Attribute | _: BoundReference => true
    case _: Alias | _: KnownNotNull | _: KnownNullable | _: GetStructField | _: SortOrder =>
      expr.children.forall(borrowedArray)
    case _ => false
  }

  private def unprotectedBatchValue(expr: Expression, capped: Boolean): Boolean = {
    val protectedBySource = capped || hasStructSubquery(expr)
    (!protectedBySource && !fitsConsumerBatch(expr) && !borrowedArray(expr)) ||
    expr.children.exists(unprotectedBatchValue(_, protectedBySource))
  }

  /** Physical accumulators and partitioners request their arguments as complete batch arrays. */
  def operatorArgumentFallbackReason(expr: Expression): Option[String] =
    if (hasStructSubqueryContext(expr) && unprotectedBatchValue(expr, capped = false)) {
      Some("Unbounded operator siblings of struct scalar subqueries require Spark")
    } else if (hasStructSubquery(expr) && !fitsConsumerBatch(expr)) {
      Some("Variable-width struct scalar subquery operator arguments require Spark")
    } else {
      None
    }

  /** Opaque JVM kernels cannot participate in the native consumer's bounded evaluation. */
  private[serde] def dispatchedConsumerFallbackReason(
      expr: Expression,
      proto: ExprOuterClass.Expr): Option[String] = {
    def containsDispatch(message: Message): Boolean = message match {
      case e: ExprOuterClass.Expr if e.hasJvmScalarUdf => true
      case other =>
        other.getAllFields.values().asScala.exists {
          case child: Message => containsDispatch(child)
          case children: java.util.List[_] =>
            children.asScala.exists {
              case child: Message => containsDispatch(child)
              case _ => false
            }
          case _ => false
        }
    }
    if (needsBoundedEvaluation(expr) && containsDispatch(proto)) {
      Some("JVM-dispatched consumers of variable-width struct scalar subqueries require Spark")
    } else {
      None
    }
  }

  /** Keep an unknown serialized child from silently disabling the native width guard. */
  private[serde] def unknownConsumerFallbackReason(
      expr: Expression,
      proto: ExprOuterClass.Expr): Option[String] = {
    import ExprOuterClass.Expr.ExprStructCase._
    def recognized(e: ExprOuterClass.Expr): Boolean = e.getExprStructCase match {
      case LITERAL | BOUND | UNBOUND | SUBQUERY | ADD | SUBTRACT | MULTIPLY | DIVIDE |
          INTEGRAL_DIVIDE | REMAINDER | UNARY_MINUS | CAST | CHECK_OVERFLOW | EQ | NEQ | GT |
          GT_EQ | LT | LT_EQ | EQNULLSAFE | NEQNULLSAFE | AND | OR | NOT | IS_NULL | IS_NOT_NULL |
          LIKE | RLIKE | SCALARFUNC | BITWISEAND | BITWISEOR | BITWISEXOR | BITWISESHIFTLEFT |
          BITWISESHIFTRIGHT | CASEWHEN | IN | IF | SORT_ORDER | NORMALIZE_NAN_AND_ZERO |
          BLOOM_FILTER_MIGHT_CONTAIN | CREATE_NAMED_STRUCT | GET_STRUCT_FIELD | TO_JSON |
          TO_PRETTY_STRING | LIST_EXTRACT | GET_ARRAY_STRUCT_FIELDS | ARRAY_INSERT | FROM_JSON |
          TO_CSV | ARRAYS_ZIP | PRECISE_TIMESTAMP_CONVERSION | AT_LEAST_N_NON_NULLS | HOUR |
          MINUTE | SECOND | UNIX_TIMESTAMP | HOURS_TRANSFORM =>
        true
      case _ => false
    }
    def containsUnknown(message: Message): Boolean = {
      val unknown = message match {
        case e: ExprOuterClass.Expr => !recognized(e)
        case _ => false
      }
      unknown || message.getAllFields.values().asScala.exists {
        case child: Message => containsUnknown(child)
        case children: java.util.List[_] =>
          children.asScala.exists {
            case child: Message => containsUnknown(child)
            case _ => false
          }
        case _ => false
      }
    }
    if (needsBoundedEvaluation(expr) && containsUnknown(proto)) {
      Some("Unreviewed consumers of expanded struct scalar subqueries require Spark")
    } else {
      None
    }
  }

  /** Input slicing for one projected scalar also changes its independent siblings' batches. */
  def projectionConsumerFallbackReason(
      expressions: Seq[Expression],
      protos: Seq[ExprOuterClass.Expr]): Option[String] = {
    if (expressions.exists(hasStructSubqueryContext) &&
      expressions.exists(unprotectedBatchValue(_, capped = false))) {
      return Some("Unbounded projection siblings of struct scalar subqueries require Spark")
    }
    val source = expressions.find(e => hasStructSubquery(e) && !fitsConsumerBatch(e))
    source.flatMap { context =>
      if (expressions.exists(!_.deterministic)) {
        Some("Nondeterministic siblings of variable-width struct scalar subqueries require Spark")
      } else {
        protos.iterator
          .flatMap { proto =>
            // The materialized scalar controls slicing for every sibling in this projection.
            dispatchedConsumerFallbackReason(context, proto)
              .orElse(unknownConsumerFallbackReason(context, proto))
          }
          .take(1)
          .toSeq
          .headOption
      }
    }
  }

  private def isTimestamp(dt: DataType): Boolean =
    dt == TimestampType || dt == TimestampNTZType

  private def isEpochCast(from: DataType, to: DataType): Boolean =
    (isTimestamp(from) && to == LongType) || (from == LongType && isTimestamp(to))

  private def hasStructSubquerySource(expr: Expression, inputs: Set[ExprId]): Boolean =
    expr.references.exists(a => inputs.contains(a.exprId)) ||
      (expr.containsPattern(PLAN_EXPRESSION) && expr.exists {
        case subquery: ExecSubqueryExpression => subquery.dataType.isInstanceOf[StructType]
        case _ => false
      })

  private def hasStructSubquery(expr: Expression): Boolean =
    hasStructSubquerySource(expr, currentStructSubqueryInputs.value)

  private def hasStructSubqueryContext(expr: Expression): Boolean =
    currentPlanHasStructSubquery.value || hasStructSubquery(expr)

  private def clockFallbackReason(
      child: Expression,
      timeZoneId: Option[String]): Option[String] = {
    if (child.dataType == TimestampNTZType || CometTimeZone.isFixedOffset(timeZoneId)) {
      None
    } else {
      Some(timezoneConsumerFallbackReason)
    }
  }

  // Pure value reads can run on rows Spark skips; deterministic computations can still fail.
  private val primitiveNumericTypes: Set[DataType] =
    Set(ByteType, ShortType, IntegerType, LongType, FloatType, DoubleType)

  private def positiveIntegralLiteral(expr: Expression): Boolean = expr match {
    case Literal(value: Number, dt)
        if Set[DataType](ByteType, ShortType, IntegerType, LongType).contains(dt) =>
      value.longValue() > 0
    case cast: Cast if Cast.canUpCast(cast.child.dataType, cast.dataType) =>
      positiveIntegralLiteral(cast.child)
    case _ => false
  }

  private def canEvaluateForSkippedRows(expr: Expression): Boolean = expr match {
    case _: Literal | _: Attribute | _: BoundReference | _: ExecSubqueryExpression => true
    case field: GetStructField => canEvaluateForSkippedRows(field.child)
    case _: UnixDate | _: DateFromUnixDate =>
      expr.children.forall(canEvaluateForSkippedRows)
    case _: IsNull | _: IsNotNull | _: Not | _: And | _: Or | _: EqualTo | _: EqualNullSafe |
        _: LessThan | _: LessThanOrEqual | _: GreaterThan | _: GreaterThanOrEqual | _: If |
        _: CaseWhen =>
      expr.children.forall(canEvaluateForSkippedRows)
    case remainder: Remainder
        if Set[DataType](ByteType, ShortType, IntegerType, LongType)
          .contains(remainder.left.dataType) && positiveIntegralLiteral(remainder.right) =>
      canEvaluateForSkippedRows(remainder.left)
    case _: XxHash64 | _: Murmur3Hash
        if expr.children.forall(child => primitiveNumericTypes.contains(child.dataType)) =>
      expr.children.forall(canEvaluateForSkippedRows)
    // Spark initializes the Bloom filter while generating code, even below a skipped Boolean arm.
    case bloom: BloomFilterMightContain if QueryPlanSerde.usesGeneratedExpressionEvaluation =>
      canEvaluateForSkippedRows(bloom.right)
    case hour: Hour if clockFallbackReason(hour.child, hour.timeZoneId).isEmpty =>
      canEvaluateForSkippedRows(hour.child)
    case minute: Minute if clockFallbackReason(minute.child, minute.timeZoneId).isEmpty =>
      canEvaluateForSkippedRows(minute.child)
    case second: Second if clockFallbackReason(second.child, second.timeZoneId).isEmpty =>
      canEvaluateForSkippedRows(second.child)
    case cast: Cast
        if primitiveNumericTypes.contains(cast.child.dataType) &&
          primitiveNumericTypes.contains(cast.dataType) &&
          Cast.canUpCast(cast.child.dataType, cast.dataType) =>
      canEvaluateForSkippedRows(cast.child)
    case _ => false
  }

  private def formatsWithoutErrors(dt: DataType): Boolean = dt match {
    case StringType | BooleanType | NullType => true
    case _: DecimalType => true
    case _ => primitiveNumericTypes.contains(dt)
  }

  // A vectorized consumer can visit a later row before Spark asks for it, even under LIMIT.
  // NULL masks alone do not prove that all other rows/arguments are free of errors.
  // Existing byte/shape gates still govern allocation failures; this checks SQL evaluation.
  private def canEvaluateInAnyRowOrder(expr: Expression): Boolean =
    canEvaluateForSkippedRows(expr) || (expr match {
      case _: Alias | _: KnownNotNull | _: KnownNullable | _: KnownFloatingPointNormalized |
          _: NormalizeNaNAndZero | _: IsNaN | _: SortOrder | _: GetStructField | _: If |
          _: CaseWhen | _: Coalesce | _: Greatest | _: Least | _: CreateNamedStruct |
          _: CreateArray | _: Concat | _: ConcatWs | _: Length | _: OctetLength | _: Ascii |
          _: Lower | _: Upper | _: XxHash64 | _: Murmur3Hash | _: Reverse | _: IsNull |
          _: IsNotNull | _: Not | _: And | _: Or | _: EqualTo | _: EqualNullSafe | _: LessThan |
          _: LessThanOrEqual | _: GreaterThan | _: GreaterThanOrEqual | _: In | _: InSet |
          _: ArrayContains | _: ArrayPosition | _: ArrayRemove | _: ArraysOverlap |
          _: ArrayUnion | _: GetMapValue | _: GetArrayStructFields | _: Size | _: ArraysZip |
          _: Log | _: Log2 | _: Log10 | _: Rand =>
        expr.children.forall(canEvaluateInAnyRowOrder)
      case _: StringLPad | _: StringRPad =>
        boundedResultBytes(expr).exists(_ <= Int.MaxValue) &&
        expr.children.forall(canEvaluateInAnyRowOrder)
      case _: StringRepeat =>
        boundedResultBytes(expr).exists(_ <= Int.MaxValue) &&
        expr.children.forall(canEvaluateInAnyRowOrder)
      case _: UnixDate | _: DateFromUnixDate | _: DayOfWeek | _: WeekDay =>
        expr.children.forall(canEvaluateInAnyRowOrder)
      case hour: Hour if clockFallbackReason(hour.child, hour.timeZoneId).isEmpty =>
        canEvaluateInAnyRowOrder(hour.child)
      case minute: Minute if clockFallbackReason(minute.child, minute.timeZoneId).isEmpty =>
        canEvaluateInAnyRowOrder(minute.child)
      case second: Second if clockFallbackReason(second.child, second.timeZoneId).isEmpty =>
        canEvaluateInAnyRowOrder(second.child)
      case unix: UnixTimestamp if isTimestamp(unix.timeExp.dataType) =>
        unix.children.forall(canEvaluateInAnyRowOrder)
      case unix: ToUnixTimestamp if isTimestamp(unix.timeExp.dataType) =>
        unix.children.forall(canEvaluateInAnyRowOrder)
      case conversion: PreciseTimestampConversion
          if isEpochCast(conversion.fromType, conversion.toType) =>
        conversion.children.forall(canEvaluateInAnyRowOrder)
      case cast: Cast
          if cast.child.dataType == cast.dataType ||
            isEpochCast(cast.child.dataType, cast.dataType) ||
            (cast.child.dataType.isInstanceOf[DecimalType] &&
              Set[DataType](FloatType, DoubleType).contains(cast.dataType)) ||
            (cast.evalMode != EvalMode.ANSI && primitiveNumericTypes.contains(cast.dataType) &&
              (primitiveNumericTypes.contains(cast.child.dataType) ||
                cast.child.dataType == StringType)) ||
            (primitiveNumericTypes.contains(cast.child.dataType) &&
              primitiveNumericTypes.contains(cast.dataType) &&
              Cast.canUpCast(cast.child.dataType, cast.dataType)) ||
            (cast.dataType == StringType && formatsWithoutErrors(cast.child.dataType)) =>
        canEvaluateInAnyRowOrder(cast.child)
      case arithmetic: Add
          if primitiveNumericTypes.contains(arithmetic.dataType) &&
            arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case arithmetic: Subtract
          if primitiveNumericTypes.contains(arithmetic.dataType) &&
            arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case arithmetic: Multiply
          if primitiveNumericTypes.contains(arithmetic.dataType) &&
            arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case arithmetic: Divide if arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case arithmetic: IntegralDivide if arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case arithmetic: Remainder if arithmetic.evalMode != EvalMode.ANSI =>
        arithmetic.children.forall(canEvaluateInAnyRowOrder)
      case _ => false
    })

  def broadcastBuildKeysFallbackReason(keys: Seq[Expression]): Option[String] =
    if (keys.exists(hasStructSubqueryContext) && keys.exists { key =>
        !canEvaluateInAnyRowOrder(key) || key.exists(_.isInstanceOf[BloomFilterMightContain])
      }) {
      // Spark hashes the build keys while broadcasting; native tasks may never run on an
      // empty streamed RDD. Bloom deserialization is also required at that initialization.
      Some("Fallible struct scalar subquery broadcast keys require Spark build evaluation")
    } else {
      None
    }

  def topKInitializationFallbackReason(expressions: Seq[Expression]): Option[String] =
    if (expressions.exists(hasStructSubqueryContext) &&
      expressions.exists(_.exists(_.isInstanceOf[BloomFilterMightContain]))) {
      // TopK builds its ordering and collect projection on the driver, even with no tasks.
      Some(bloomInitializationFallbackReason)
    } else {
      None
    }

  def groupedAggregateFallbackReason(plan: BaseAggregateExec): Option[String] =
    if (plan.groupingExpressions.nonEmpty &&
      plan.aggregateExpressions.exists(hasStructSubqueryContext) &&
      plan.aggregateExpressions.exists(_.aggregateFunction match {
        case sum: Sum =>
          sum.dataType.isInstanceOf[DecimalType] &&
          CometEvalModeUtil.sumEvalMode(sum) == EvalMode.ANSI
        case _ => false
      })) {
      // Spark defers decimal SUM overflow until final evaluation; the native grouped
      // accumulator raises during update, ahead of other rows/aggregates' errors.
      Some("ANSI grouped decimal SUM in struct scalar subquery plans requires Spark evaluation")
    } else {
      None
    }

  private def skippedObservableArguments(expr: Expression): Seq[Expression] =
    expr.children.zipWithIndex.collect {
      case (child, index)
          if expr.children.take(index).exists(_.nullable) &&
            !canEvaluateForSkippedRows(child) =>
        child
    }

  private def hasSkippedObservableArgument(expr: Expression): Boolean =
    skippedObservableArguments(expr).nonEmpty

  // Keep the existing native masks, including their operator-wide sharing authorization.
  private def isNativeNullShortCircuitFamily(expr: Expression): Boolean = expr match {
    case _: EqualTo | _: LessThan | _: LessThanOrEqual | _: GreaterThan | _: GreaterThanOrEqual |
        _: ArrayContains | _: ArrayRemove | _: ArrayPosition | _: ArraysOverlap | _: ArrayUnion |
        _: Slice | _: GetArrayItem | _: GetMapValue | _: StringLPad | _: StringRPad |
        _: BloomFilterMightContain =>
      true
    case _ => false
  }

  private def hasNativeNullShortCircuit(expr: Expression): Boolean =
    isNativeNullShortCircuitFamily(expr) &&
      (QueryPlanSerde.canShortCircuitNulls(expr) ||
        skippedObservableArguments(expr).forall(QueryPlanSerde.isEagerSharedSubexpression))

  // Opaque/stateful siblings prevent native projection fusion. Keep their wide producer on rows
  // too; conservatively include native-opt-in handlers without duplicating dispatch decisions.
  private def requiresRowProjection(expr: Expression): Boolean =
    !expr.deterministic || expr.exists {
      case _: ScalaUDF | _: LambdaFunction | _: StaticInvoke | _: RegExpExtract |
          _: RegExpExtractAll | _: DateFormatClass =>
        true
      case node =>
        QueryPlanSerde.exprSerdeMap.get(node.getClass).forall { handler =>
          val serde = handler.asInstanceOf[CometExpressionSerde[Expression]]
          !CometConf.isExprEnabled(serde.getExprConfigName(node)) || (handler match {
            case _: CometCodegenDispatch[_] => true
            case _: NativeOptInAvailable if !handler.isInstanceOf[CodegenDispatchFallback] => true
            case _ =>
              serde.getSupportLevel(node) match {
                case Compatible(_, None) => false
                case _ => true
              }
          })
        }
    }

  // These serdes repeat operands in synthetic NULL/domain guards, even when non-nullable.
  private def repeatsStatefulOperand(expr: Expression): Boolean = expr match {
    case coalesce: Coalesce => coalesce.children.dropRight(1).exists(!_.deterministic)
    case _: Size | _: ArraysZip | _: MapFromArrays | _: Log | _: Log10 | _: Log2 =>
      !expr.deterministic
    case division: Divide if division.evalMode != EvalMode.ANSI =>
      !division.right.deterministic
    case division: IntegralDivide if division.evalMode != EvalMode.ANSI =>
      !division.right.deterministic
    case remainder: Remainder if remainder.evalMode != EvalMode.ANSI =>
      !remainder.right.deterministic
    case _ => false
  }

  private def evaluationFallbackReason(expr: Expression): Option[String] = {
    if (!hasStructSubqueryContext(expr)) return None
    expr match {
      // Runtime masks cannot defer native Bloom setup. FALLBACK can also recover from a
      // code-generation error and interpret the expression, including its first-error order.
      // Whole-stage codegen initializes on the driver, even when there are no input tasks.
      case _
          if expr.exists(_.isInstanceOf[BloomFilterMightContain]) &&
            (!SQLConf.get
              .getConfString("spark.sql.codegen.factoryMode", "FALLBACK")
              .equalsIgnoreCase("CODEGEN_ONLY") ||
              SQLConf.get.wholeStageEnabled) =>
        Some(bloomInitializationFallbackReason)
      // IN skips candidates after a NULL value or a matching candidate. Native dynamic IN only
      // stops after a batch-wide match, and Spark may evaluate shared candidates eagerly.
      case in: In
          if in.list.exists(!_.isInstanceOf[Literal]) &&
            (in.references.nonEmpty || !QueryPlanSerde.canSkipEvaluation(in.list) ||
              !in.deterministic || (in.value.nullable &&
                in.list.exists(candidate => !canEvaluateForSkippedRows(candidate)))) =>
        Some("Nested membership of struct scalar subqueries requires Spark")
      case and: And if !canEvaluateForSkippedRows(and.right) =>
        Some("Boolean short-circuit consumers of struct scalar subqueries require Spark")
      case or: Or if !canEvaluateForSkippedRows(or.right) =>
        Some("Boolean short-circuit consumers of struct scalar subqueries require Spark")
      // Spark evaluates the divisor first, and may skip the numerator for a NULL or legacy zero.
      case division: DivModLike if !canEvaluateForSkippedRows(division.left) =>
        Some("Ordered arithmetic consumers of struct scalar subqueries require Spark")
      case modulo: Pmod if !canEvaluateForSkippedRows(modulo.left) =>
        Some("Ordered arithmetic consumers of struct scalar subqueries require Spark")
      // Generated string concat skips after NULL; interpreted string concat evaluates every arg.
      case _: Concat if hasSkippedObservableArgument(expr) =>
        Some("Ordered concat consumers of struct scalar subqueries require Spark")
      // Spark evaluates concat_ws arguments even when its separator is NULL. The native
      // literal-NULL rewrite must not erase an error or stateful evaluation in those arguments.
      case concat: ConcatWs
          if hasLiteralNullSeparator(concat) &&
            (hasSkippedObservableArgument(concat) ||
              concat.exists(_.isInstanceOf[BloomFilterMightContain])) =>
        Some("NULL-separator concat_ws struct scalar subquery arguments require Spark evaluation")
      // Round's NULL-scale rewrite also erases child code generation and its Bloom setup.
      case round: Round
          if QueryPlanSerde.usesGeneratedExpressionEvaluation && round.scale.nullable &&
            round.child.exists(_.isInstanceOf[BloomFilterMightContain]) =>
        Some("Nullable-scale round initialization of struct scalar subqueries requires Spark")
      case _
          if (isNullIntolerant(expr) || isNativeNullShortCircuitFamily(expr)) &&
            hasSkippedObservableArgument(expr) && !hasNativeNullShortCircuit(expr) =>
        Some("Nullable ordered consumers of struct scalar subqueries require Spark")
      case _ => None
    }
  }

  /**
   * Admit only reviewed consumers of dates and timestamps on newly supported struct-subquery
   * paths. Unknown consumers stay on Spark by default, including expressions whose result is not
   * a calendar type. Transport, comparisons and epoch arithmetic do not interpret a calendar.
   *
   * Check the whole expression before serialization or codegen dispatch, including wrappers and
   * whole-struct casts. A physical subquery may be reused, but it is still an
   * [[ExecSubqueryExpression]]; its result must never be evaluated while planning this check.
   */
  private[serde] def consumerFallbackReason(expr: Expression): Option[String] =
    specificConsumerFallbackReason(expr).orElse {
      // Constants can fail during setup. Check every subtree, including Bloom's filter input.
      if (hasStructSubqueryContext(expr) && expr.exists { node =>
          hasStructSubqueryContext(node) && !canEvaluateInAnyRowOrder(node) &&
          !(deferredTopKSort.value && deferredTopKKey(node))
        }) {
        Some("Fallible struct scalar subquery consumers require Spark evaluation")
      } else None
    }

  private def specificConsumerFallbackReason(expr: Expression): Option[String] = {
    // Skip paths without struct subqueries. A newly admitted calendar consumer can be a
    // sibling or ancestor of a struct-producing operator without using its result directly.
    // Aliases still carry value provenance through the owning plan's ExprIds.
    val derivedInput =
      expr.references.exists(a => currentStructSubqueryInputs.value.contains(a.exprId))
    if (!currentPlanHasStructSubquery.value && !derivedInput &&
      (!expr.containsPattern(PLAN_EXPRESSION) || !ExecSubqueryExpression.hasSubquery(expr))) {
      return None
    }
    if (hasStructSubqueryContext(expr) && repeatsStatefulOperand(expr)) {
      return Some(
        "Repeated stateful evaluation of struct scalar subquery consumers requires Spark")
    }
    val selectsValue = expr match {
      case _: Attribute | _: BoundReference | _: Literal | _: ExecSubqueryExpression |
          _: GetStructField | _: Alias | _: KnownNotNull | _: KnownNullable | _: If |
          _: CaseWhen | _: Coalesce | _: Greatest | _: Least | _: CreateNamedStruct |
          _: SortOrder =>
        true
      case _ => false
    }
    val unboundedConstructor = hasStructSubqueryContext(expr) && !selectsValue &&
      fixedTypeBytes(expr.dataType).isEmpty &&
      !boundedResultBytes(expr).exists(_ <= Int.MaxValue)

    // Native consumers may change batch boundaries to avoid repeating a large scalar field.
    // Stateful consumers must retain Spark's evaluation order and their pre-existing fallback.
    if (!expr.deterministic && needsBoundedEvaluation(expr)) {
      return Some(
        "Nondeterministic consumers of variable-width struct scalar subqueries require Spark")
    }

    val touchesCalendar =
      containsCalendarType(expr.dataType) ||
        expr.children.exists(child => containsCalendarType(child.dataType))

    val reason = expr match {
      // Preserve the existing guard for plain scalar date subqueries as well as struct fields.
      case unix: UnixTimestamp
          if unix.timeExp.dataType == DateType &&
            (ExecSubqueryExpression.hasSubquery(unix.timeExp) || hasStructSubqueryContext(
              unix.timeExp)) =>
        Some(dateUnixTimestampFallbackReason)
      // Retain timezone protection for plain scalar subqueries too.
      case hour: Hour => clockFallbackReason(hour.child, hour.timeZoneId)
      case minute: Minute => clockFallbackReason(minute.child, minute.timeZoneId)
      case second: Second => clockFallbackReason(second.child, second.timeZoneId)
      case _ if !hasStructSubqueryContext(expr) || !touchesCalendar => None
      // These nodes only carry, select, compare or inspect the validity of raw values.
      // Keep the list explicit: a new date-consuming expression must be reviewed before admission.
      case _: Literal | _: Attribute | _: BoundReference | _: ScalarSubquery | _: Alias |
          _: GetStructField | _: CreateNamedStruct | _: CreateArray | _: CreateMap |
          _: GetArrayItem | _: GetArrayStructFields | _: GetMapValue | _: ElementAt | _: If |
          _: CaseWhen | _: Coalesce | _: Greatest | _: Least | _: KnownNotNull |
          _: KnownNullable | _: TryEval | _: SortOrder | _: IsNull | _: IsNotNull | _: In |
          _: InSet | _: EqualTo | _: EqualNullSafe | _: LessThan | _: LessThanOrEqual |
          _: GreaterThan | _: GreaterThanOrEqual =>
        None
      case cast: Cast
          if cast.child.dataType == cast.dataType ||
            isEpochCast(cast.child.dataType, cast.dataType) =>
        None
      case conversion: PreciseTimestampConversion
          if isEpochCast(conversion.fromType, conversion.toType) =>
        None
      case _: UnixDate | _: DateFromUnixDate => None
      // These two date fields use epoch-day arithmetic rather than chrono's calendar.
      case _: DayOfWeek | _: WeekDay => None
      // Timestamp-to-epoch conversions are arithmetic; dates first need calendar conversion.
      case unix: UnixTimestamp if isTimestamp(unix.timeExp.dataType) => None
      case unix: ToUnixTimestamp if isTimestamp(unix.timeExp.dataType) => None
      case _ => Some(calendarConsumerFallbackReason)
    }

    // An allowed wrapper must not hide an unreviewed consumer in its children.
    reason
      .orElse {
        if (unboundedConstructor) {
          Some("Unbounded struct scalar subquery constructors require Spark row evaluation")
        } else None
      }
      .orElse {
        expr.children.iterator.map(specificConsumerFallbackReason).collectFirst {
          case Some(childReason) =>
            childReason
        }
      }
      .orElse(evaluationFallbackReason(expr))
  }

  override def getUnsupportedReasons(): Seq[String] = Seq(
    "Not all data types are supported for scalar subquery results",
    "Struct fields must have supported types and distinct names at every nesting level")

  // The shared type gate handles general capabilities. Only the Arrow IPC scalar-subquery
  // bridge's additional struct-only and decimal-scale restrictions belong here.
  private def supportedStructShape(dt: DataType): Boolean = dt match {
    case s: StructType => s.fields.forall(f => supportedStructShape(f.dataType))
    case _: ArrayType | _: MapType => false
    case d: DecimalType => d.scale >= 0 && d.scale <= d.precision
    case _ => true
  }

  override def getSupportLevel(expr: ScalarSubquery): SupportLevel = {
    val supported = expr.dataType match {
      case s: StructType =>
        supportedDataType(
          s,
          allowComplex = true,
          allowIntervals = false,
          allowCalendarInterval = false,
          allowTimeType = false,
          allowAnyStringType = false,
          allowDuplicateStructFieldNames = false) && supportedStructShape(s)
      case dt => supportedDataType(dt)
    }
    if (supported) {
      Compatible()
    } else {
      Unsupported(Some(s"Unsupported data type: ${expr.dataType}"))
    }
  }

  @scala.annotation.tailrec
  private def materializedSubqueryId(plan: BaseSubqueryExec): Long = plan match {
    case reused: ReusedSubqueryExec => materializedSubqueryId(reused.child)
    case other => other.id.toLong
  }

  override def convert(
      expr: ScalarSubquery,
      inputs: Seq[Attribute],
      binding: Boolean): Option[ExprOuterClass.Expr] = {
    // getSupportLevel has already checked value-transfer support. Type serialization can still
    // decline, so keep this separate check.
    val dataType = serializeDataType(expr.dataType)
    if (dataType.isEmpty) {
      withFallbackReason(
        expr,
        s"Failed to serialize datatype ${expr.dataType} for scalar subquery")
      return None
    }

    val builder = ExprOuterClass.Subquery
      .newBuilder()
      .setId(expr.exprId.id)
      .setDatatype(dataType.get)
    if (expr.dataType.isInstanceOf[StructType]) {
      builder.setStructResultId(materializedSubqueryId(expr.plan))
    }
    Some(ExprOuterClass.Expr.newBuilder().setSubquery(builder).build())
  }
}
