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

import java.time.ZoneOffset

import scala.util.Try

import org.apache.spark.sql.catalyst.expressions.{Attribute, BoundReference, CaseWhen, Cast, Coalesce, Expression, FromUTCTimestamp, If, PreciseTimestampConversion, SecondsToTimestamp, ToUTCTimestamp, TruncTimestamp}
import org.apache.spark.sql.catalyst.util.DateTimeUtils
import org.apache.spark.sql.types.{TimestampNTZType, TimestampType}

/**
 * Converts the timezone Spark stamps on an expression into an ID that native code can parse.
 *
 * Spark resolves timezone IDs with `ZoneId.of(id, ZoneId.SHORT_IDS)`, which accepts forms such as
 * `Z`, `+8`, `+08:00:00`, `GMT+8` and `PST`. Native code parses them with arrow's `Tz`, which
 * accepts only IANA zone names and offsets written as `+HH`, `+HHMM` or `+HH:MM`.
 */
object CometTimeZone {

  /**
   * The ID to pass to native code for `timeZoneId`, or None when native code cannot represent the
   * zone, which is the case for an offset with seconds. A fixed offset becomes `+HH:MM`, a zero
   * offset becomes `UTC`, and a short ID becomes its region. An expression with no timezone gets
   * `UTC`: Spark leaves it unset only on casts that do not use it.
   */
  def nativeId(timeZoneId: Option[String]): Option[String] = timeZoneId match {
    case None => Some("UTC")
    case Some(id) =>
      Try(DateTimeUtils.getZoneId(id).normalized()).toOption.flatMap {
        case offset: ZoneOffset if offset.getTotalSeconds == 0 => Some("UTC")
        case offset: ZoneOffset if offset.getTotalSeconds % 60 == 0 => Some(offset.getId)
        case _: ZoneOffset => None
        case region => Some(region.getId)
      }
  }

  /** Whether `timeZoneId` has a zero offset at every instant. */
  def isUtc(timeZoneId: Option[String]): Boolean = nativeId(timeZoneId).contains("UTC")

  /** Whether native code can represent a timezone whose offset never changes. */
  private[serde] def isFixedOffset(timeZoneId: Option[String]): Boolean =
    nativeId(timeZoneId).exists { id =>
      Try(DateTimeUtils.getZoneId(id).getRules.isFixedOffset).getOrElse(false)
    }

  private[serde] val clockInputUnsupportedReason: String =
    "Clock expressions with inputs that may be scalar require Spark for non-fixed timezones"

  /** Only value branches determine whether IF/CASE/COALESCE can return a scalar. */
  private[serde] def hasArrayResult(input: Expression): Boolean = input match {
    case _: Attribute | _: BoundReference => true
    // Other casts can be replaced with a scalar NULL by their serde.
    case cast: Cast if cast.dataType == TimestampType || cast.dataType == TimestampNTZType =>
      hasArrayResult(cast.child)
    case conditional: If =>
      hasArrayResult(conditional.trueValue) && hasArrayResult(conditional.falseValue)
    // Without an ELSE, a batch with no match returns a scalar NULL.
    case conditional: CaseWhen =>
      conditional.branches.forall { case (_, value) => hasArrayResult(value) } &&
      conditional.elseValue.exists(hasArrayResult)
    // COALESCE lowers to CASE, with the final child as its explicit ELSE.
    case coalesce: Coalesce =>
      coalesce.children.nonEmpty && coalesce.children.forall(hasArrayResult)
    case seconds: SecondsToTimestamp => hasArrayResult(seconds.child)
    case precise: PreciseTimestampConversion => hasArrayResult(precise.child)
    case trunc: TruncTimestamp =>
      hasArrayResult(trunc.timestamp) || hasArrayResult(trunc.format)
    case from: FromUTCTimestamp => hasArrayResult(from.left) || hasArrayResult(from.right)
    case to: ToUTCTimestamp => hasArrayResult(to.left) || hasArrayResult(to.right)
    case _ => false
  }

  private[serde] def clockSupportLevel(
      child: Expression,
      timeZoneId: Option[String]): SupportLevel = {
    if (child.dataType == TimestampNTZType || isFixedOffset(timeZoneId) || hasArrayResult(
        child)) {
      supportLevel(timeZoneId)
    } else {
      Unsupported(Some(clockInputUnsupportedReason))
    }
  }

  /** `Unsupported` when native code cannot represent `timeZoneId`, `Compatible` otherwise. */
  def supportLevel(timeZoneId: Option[String]): SupportLevel = nativeId(timeZoneId) match {
    case Some(_) => Compatible()
    case None => Unsupported(Some(unsupportedReason(timeZoneId)))
  }

  def unsupportedReason(timeZoneId: Option[String]): String =
    s"Timezone ${timeZoneId.getOrElse("")} cannot be represented in native code"
}
