# sql_gen_17_features_ts

Codes for extracting time series of 17 features used in SAPS-II, based on codes at `concepts/firstday` from the repo [mimic-code](https://github.com/MIT-LCP/mimic-code.git).

## Modification

The original codes are used for gettitng stats (max/min/avg) of the 17 features in the first day. Therefore we made two pieces of modification.

### Time Limit

We removed the limit of ending time, resulting that we could get the whole time series.

```sql
--     AND le.charttime BETWEEN (ie.intime - interval '6' hour) AND (ie.intime + interval '1' day)
    AND le.charttime >= (ie.intime - interval '6' hour)
```

### Time Slot

We changed the conditions of "group by", making sure that we could get a value for each time slot instead of only the stats of the time series.

```sql
-- GROUP BY pvt.subject_id, pvt.hadm_id, pvt.icustay_id
GROUP BY pvt.hadm_id, pvt.charttime
```

