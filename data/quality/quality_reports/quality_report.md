# Data Quality Report

## Structural nulls (percent metrics)

- Rows: **5248**
- `fg_pct NULLs` = **0.0**; structural (FGA=0) = **0.0**
- `fg3_pct NULLs` = **245.0**; structural (3PA=0) = **245.0**
- `ft_pct NULLs` = **0.0**; structural (FTA=0) = **0.0**

[pct_nulls_overall.csv](pct_nulls_overall.csv)  
[pct_nulls_by_season.csv](pct_nulls_by_season.csv)

## Bio coverage

[bio_missing_by_season.csv](bio_missing_by_season.csv)  
[bio_missing_by_cohort.csv](bio_missing_by_cohort.csv)

### Plausibility outliers (Players)

- Count: **500**

[bio_players_outliers.csv](bio_players_outliers.csv)


**Examples (first 25):**

- {'personId': 3, 'firstName': 'Grant', 'lastName': 'Long', 'height': 81.0, 'bodyWeight': 3.0, 'draftYear': 1988.0}
- {'personId': 24, 'firstName': 'Keith', 'lastName': 'Jennings', 'height': 67.0, 'bodyWeight': 160.0, 'draftYear': -1.0}
- {'personId': 30, 'firstName': 'Richard', 'lastName': 'Dumas', 'height': 79.0, 'bodyWeight': 30.0, 'draftYear': 1991.0}
- {'personId': 31, 'firstName': 'Lorenzo', 'lastName': 'Williams', 'height': 81.0, 'bodyWeight': 200.0, 'draftYear': -1.0}
- {'personId': 49, 'firstName': 'Gheorghe', 'lastName': 'Muresan', 'height': 91.0, 'bodyWeight': 303.0, 'draftYear': 1993.0}
- {'personId': 51, 'firstName': 'Mahmoud', 'lastName': 'Abdul-Rauf', 'height': 73.0, 'bodyWeight': 51.0, 'draftYear': 1990.0}
- {'personId': 55, 'firstName': 'Marty', 'lastName': 'Conlon', 'height': 82.0, 'bodyWeight': 224.0, 'draftYear': -1.0}
- {'personId': 64, 'firstName': 'Sam', 'lastName': 'Perkins', 'height': 81.0, 'bodyWeight': 64.0, 'draftYear': 1984.0}
- {'personId': 66, 'firstName': 'Marques', 'lastName': 'Bragg', 'height': 80.0, 'bodyWeight': 230.0, 'draftYear': -1.0}
- {'personId': 70, 'firstName': 'Steve', 'lastName': 'Kerr', 'height': 73.0, 'bodyWeight': 70.0, 'draftYear': 1988.0}
- {'personId': 74, 'firstName': 'Derrick', 'lastName': 'Phelps', 'height': 76.0, 'bodyWeight': 181.0, 'draftYear': -1.0}
- {'personId': 76, 'firstName': 'Cedric', 'lastName': 'Ceballos', 'height': 79.0, 'bodyWeight': 76.0, 'draftYear': 1990.0}
- {'personId': 78, 'firstName': 'LaSalle', 'lastName': 'Thompson', 'height': 82.0, 'bodyWeight': 78.0, 'draftYear': 1982.0}
- {'personId': 85, 'firstName': 'Brent', 'lastName': 'Price', 'height': 73.0, 'bodyWeight': 85.0, 'draftYear': 1992.0}
- {'personId': 100, 'firstName': 'Tim', 'lastName': 'Legler', 'height': 76.0, 'bodyWeight': 200.0, 'draftYear': -1.0}
- {'personId': 116, 'firstName': 'David', 'lastName': 'Wood', 'height': 81.0, 'bodyWeight': 230.0, 'draftYear': -1.0}
- {'personId': 123, 'firstName': 'Robert', 'lastName': 'Pack', 'height': 74.0, 'bodyWeight': 190.0, 'draftYear': -1.0}
- {'personId': 128, 'firstName': 'David', 'lastName': 'Benoit', 'height': 80.0, 'bodyWeight': 220.0, 'draftYear': -1.0}
- {'personId': 133, 'firstName': 'David', 'lastName': 'Wesley', 'height': 73.0, 'bodyWeight': 203.0, 'draftYear': -1.0}
- {'personId': 141, 'firstName': 'Anthony', 'lastName': 'Tucker', 'height': 80.0, 'bodyWeight': 141.0, 'draftYear': -22.0}
- {'personId': 154, 'firstName': 'Adrian', 'lastName': 'Caldwell', 'height': 80.0, 'bodyWeight': 154.0, 'draftYear': -22.0}
- {'personId': 160, 'firstName': 'Tim', 'lastName': 'Breaux', 'height': 79.0, 'bodyWeight': 215.0, 'draftYear': -1.0}
- {'personId': 164, 'firstName': 'Chris', 'lastName': 'Childs', 'height': 75.0, 'bodyWeight': 195.0, 'draftYear': -1.0}
- {'personId': 173, 'firstName': 'Keith', 'lastName': 'Askins', 'height': 80.0, 'bodyWeight': 173.0, 'draftYear': -22.0}
- {'personId': 176, 'firstName': 'Antonio', 'lastName': 'Harvey', 'height': 83.0, 'bodyWeight': 250.0, 'draftYear': -1.0}
