1) The filename is the variable name.

2) Each file contains data for out-of-sample forecasts. Each column denotes a cohort. For example, for variables at monthly frequency, there are 1,176 columns (not counting the first) for cohorts from 1926:01 to 2024:12. Rows correspond to dates. The in-sample data corresponds to the last column.

3) Data for all variables generally ends 2043 except
"#02 (vp) ends 2021:12"
"#03 (impvar) ends 2023:08"
"#04 (vrp) ends 2023:12"
"#07 (skew) ends 2019:2"
"#13 (sntm) ends 2023:11"
"#17 (fbm) ends 2024:11"
"#22 (rsvix) ends 2023:08"

4) Two variables (#13 sntm and #17 fbm) are PLS estimates. They need to be recomputed every period for the forecasting target, and for the frequency of forecast for OOS. There are 8 versions first; 4 for simple returns, last 4 for log returns; then predicting at monthly, quarterly, annual (ending Dec), and annual (ending Jun) frequency.

5) One variable (#22 rsix) has four varieties for predicting monthly, quarterly, annual (ending Dec), and annual (ending Jun) frequency.

