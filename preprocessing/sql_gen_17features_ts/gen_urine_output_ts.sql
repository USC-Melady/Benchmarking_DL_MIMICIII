-- ------------------------------------------------------------------
-- Purpose: Create a view of the urine output for each ICUSTAY_ID over the first 24 hours.
-- ------------------------------------------------------------------

DROP MATERIALIZED VIEW IF EXISTS mengcz_urine_output_ts CASCADE;
create materialized view mengcz_urine_output_ts as
with uo as
(
    SELECT
      -- patient identifiers
      ie.subject_id,
      ie.hadm_id,
      ie.icustay_id,
      oe.charttime

      -- volumes associated with urine output ITEMIDs
      ,
      sum(
      -- we consider input of GU irrigant as a negative volume
          CASE WHEN oe.itemid = 227488
            THEN -1 * VALUE
          ELSE VALUE END
      ) AS UrineOutput
    FROM icustays ie
      -- Join to the outputevents table to get urine output
      LEFT JOIN outputevents oe
      -- join on all patient identifiers
        ON ie.subject_id = oe.subject_id AND ie.hadm_id = oe.hadm_id AND ie.icustay_id = oe.icustay_id
           -- and ensure the data occurs during the first day
           -- and oe.charttime between ie.intime and (ie.intime + interval '1' day) -- first ICU day
           AND oe.charttime >= ie.intime -- all ICU days
    WHERE itemid IN
          (
            -- these are the most frequently occurring urine output observations in CareVue
            40055, -- "Urine Out Foley"
                   43175, -- "Urine ."
                   40069, -- "Urine Out Void"
                   40094, -- "Urine Out Condom Cath"
                   40715, -- "Urine Out Suprapubic"
                   40473, -- "Urine Out IleoConduit"
                   40085, -- "Urine Out Incontinent"
                   40057, -- "Urine Out Rt Nephrostomy"
                   40056, -- "Urine Out Lt Nephrostomy"
                   40405, -- "Urine Out Other"
                   40428, -- "Urine Out Straight Cath"
                          40086, --	Urine Out Incontinent
                          40096, -- "Urine Out Ureteral Stent #1"
                          40651, -- "Urine Out Ureteral Stent #2"

                          -- these are the most frequently occurring urine output observations in MetaVision
                          226559, -- "Foley"
                          226560, -- "Void"
                          226561, -- "Condom Cath"
                          226584, -- "Ileoconduit"
                          226563, -- "Suprapubic"
                          226564, -- "R Nephrostomy"
                          226565, -- "L Nephrostomy"
            226567, --	Straight Cath
            226557, -- R Ureteral Stent
            226558, -- L Ureteral Stent
            227488, -- GU Irrigant Volume In
            227489  -- GU Irrigant/Urine Volume Out
          )
    GROUP BY ie.subject_id, ie.hadm_id, ie.icustay_id, oe.charttime
    ORDER BY ie.subject_id, ie.hadm_id, ie.icustay_id, oe.charttime
)
select hadm_id, charttime, avg(UrineOutput) as UrineOutput from uo
group by hadm_id, charttime
order by hadm_id, charttime;