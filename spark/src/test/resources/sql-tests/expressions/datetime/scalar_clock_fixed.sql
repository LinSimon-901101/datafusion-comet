-- Licensed to the Apache Software Foundation (ASF) under one
-- or more contributor license agreements.  See the NOTICE file
-- distributed with this work for additional information
-- regarding copyright ownership.  The ASF licenses this file
-- to you under the Apache License, Version 2.0 (the
-- "License"); you may not use this file except in compliance
-- with the License.  You may obtain a copy of the License at
--
--   http://www.apache.org/licenses/LICENSE-2.0
--
-- Unless required by applicable law or agreed to in writing,
-- software distributed under the License is distributed on an
-- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
-- KIND, either express or implied.  See the License for the
-- specific language governing permissions and limitations
-- under the License.

-- ConfigMatrix: spark.sql.session.timeZone=UTC,+01:00,-03:30
-- Config: spark.sql.adaptive.enabled=false
-- Config: spark.sql.execution.reuseSubquery=false
-- Config: spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS
-- Config: spark.sql.parquet.datetimeRebaseModeInWrite=CORRECTED
-- Config: spark.sql.parquet.datetimeRebaseModeInRead=CORRECTED
-- ConstantFolding: enabled

statement
CREATE TABLE scalar_clock_fixed(id INT, ts TIMESTAMP, ntz TIMESTAMP_NTZ) USING parquet

statement
INSERT INTO scalar_clock_fixed VALUES
  (1, timestamp_micros(9223372036854775807L), TIMESTAMP_NTZ '2024-01-15 12:30:45'),
  (2, timestamp_micros(-9223372036854775808L), TIMESTAMP_NTZ '1969-12-31 23:59:59.999999'),
  (3, timestamp_micros(0L), NULL),
  (4, NULL, NULL)

-- Plain subqueries also return runtime scalars with normal constant folding enabled.
-- Subquery reuse is disabled to keep this independent of struct subquery support.
query expect_native(hour,minute,second,unix_timestamp)
SELECT id, hour((SELECT max(ts) FROM scalar_clock_fixed)),
       minute((SELECT max(ts) FROM scalar_clock_fixed)),
       second((SELECT max(ts) FROM scalar_clock_fixed)),
       unix_timestamp((SELECT max(ts) FROM scalar_clock_fixed))
FROM scalar_clock_fixed

query expect_native(hour,minute,second,unix_timestamp)
SELECT id, hour((SELECT max(ts) FROM scalar_clock_fixed WHERE id < 0)),
       minute((SELECT max(ts) FROM scalar_clock_fixed WHERE id < 0)),
       second((SELECT max(ts) FROM scalar_clock_fixed WHERE id < 0)),
       unix_timestamp((SELECT max(ts) FROM scalar_clock_fixed WHERE id < 0))
FROM scalar_clock_fixed

-- Every row selects a scalar branch; references alone do not promise an array.
query expect_native(hour,minute,second)
SELECT id, hour(IF(id > 0, (SELECT max(ts) FROM scalar_clock_fixed), ts)),
       minute(IF(id > 0, (SELECT max(ts) FROM scalar_clock_fixed), ts)),
       second(IF(id > 0, (SELECT max(ts) FROM scalar_clock_fixed), ts))
FROM scalar_clock_fixed

-- Mixed rows return an array. Full-range fixed-offset arithmetic must cover both shapes.
query expect_native(hour,minute,second)
SELECT id, hour(CASE WHEN id = 1 THEN (SELECT max(ts) FROM scalar_clock_fixed) ELSE ts END),
       minute(CASE WHEN id = 1 THEN (SELECT max(ts) FROM scalar_clock_fixed) ELSE ts END),
       second(CASE WHEN id = 1 THEN (SELECT max(ts) FROM scalar_clock_fixed) ELSE ts END)
FROM scalar_clock_fixed

query expect_native(hour,minute,second,unix_timestamp)
SELECT id, hour((SELECT max(ntz) FROM scalar_clock_fixed)),
       minute((SELECT max(ntz) FROM scalar_clock_fixed)),
       second((SELECT max(ntz) FROM scalar_clock_fixed)),
       unix_timestamp((SELECT max(ntz) FROM scalar_clock_fixed))
FROM scalar_clock_fixed
