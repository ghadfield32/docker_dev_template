# NBA SQLite Data Catalog

## Tables
- Games
- LeagueSchedule24_25
- PlayerStatistics
- Players
- TeamHistories
- TeamStatistics

## Games

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | gameId | INTEGER | 0 | 0 |
| 1 | gameDate | TEXT | 0 | 0 |
| 2 | hometeamCity | TEXT | 0 | 0 |
| 3 | hometeamName | TEXT | 0 | 0 |
| 4 | hometeamId | INTEGER | 0 | 0 |
| 5 | awayteamCity | TEXT | 0 | 0 |
| 6 | awayteamName | TEXT | 0 | 0 |
| 7 | awayteamId | INTEGER | 0 | 0 |
| 8 | homeScore | INTEGER | 0 | 0 |
| 9 | awayScore | INTEGER | 0 | 0 |
| 10 | winner | INTEGER | 0 | 0 |
| 11 | gameType | TEXT | 0 | 0 |
| 12 | attendance | REAL | 0 | 0 |
| 13 | arenaId | INTEGER | 0 | 0 |
| 14 | gameLabel | TEXT | 0 | 0 |
| 15 | gameSubLabel | TEXT | 0 | 0 |
| 16 | seriesGameNumber | REAL | 0 | 0 |

**Profile**
- rows: 71879
- date range: 1946-11-26 19:00:00 → 2025-06-22 20:00:00
- distinct (first 5 cols): {'gameId': 71879, 'gameDate': 33977, 'hometeamCity': 46, 'hometeamName': 39, 'hometeamId': 30}

**Sample (3 rows)**

|   gameId | gameDate            | hometeamCity   | hometeamName   |   hometeamId | awayteamCity   | awayteamName   |   awayteamId |   homeScore |   awayScore |     winner | gameType   |   attendance |   arenaId | gameLabel   | gameSubLabel   |   seriesGameNumber |
|---------:|:--------------------|:---------------|:---------------|-------------:|:---------------|:---------------|-------------:|------------:|------------:|-----------:|:-----------|-------------:|----------:|:------------|:---------------|-------------------:|
| 42400407 | 2025-06-22 20:00:00 | Oklahoma City  | Thunder        |   1610612760 | Indiana        | Pacers         |   1610612754 |         103 |          91 | 1610612760 | Playoffs   |        18203 |   1000052 | NBA Finals  | Game 7         |                  7 |
| 42400406 | 2025-06-19 20:30:00 | Indiana        | Pacers         |   1610612754 | Oklahoma City  | Thunder        |   1610612760 |         108 |          91 | 1610612754 | Playoffs   |        17274 |   1000063 | NBA Finals  | Game 6         |                  6 |
| 42400405 | 2025-06-16 20:30:00 | Oklahoma City  | Thunder        |   1610612760 | Indiana        | Pacers         |   1610612754 |         120 |         109 | 1610612760 | Playoffs   |        18203 |   1000052 | NBA Finals  | Game 5         |                  5 |

## LeagueSchedule24_25

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | gameId | INTEGER | 0 | 0 |
| 1 | gameDateTimeEst | TEXT | 0 | 0 |
| 2 | gameDay | TEXT | 0 | 0 |
| 3 | arenaCity | TEXT | 0 | 0 |
| 4 | arenaState | TEXT | 0 | 0 |
| 5 | arenaName | TEXT | 0 | 0 |
| 6 | gameLabel | TEXT | 0 | 0 |
| 7 | gameSubLabel | TEXT | 0 | 0 |
| 8 | gameSubtype | TEXT | 0 | 0 |
| 9 | gameSequence | INTEGER | 0 | 0 |
| 10 | seriesGameNumber | REAL | 0 | 0 |
| 11 | seriesText | TEXT | 0 | 0 |
| 12 | weekNumber | INTEGER | 0 | 0 |
| 13 | hometeamId | INTEGER | 0 | 0 |
| 14 | awayteamId | INTEGER | 0 | 0 |

**Profile**
- rows: 1408
- date range: 2024-10-04 12:00:00+00:00 → 2025-10-17T22:30:00Z
- distinct (first 5 cols): {'gameId': 1408, 'gameDateTimeEst': 914, 'gameDay': 7, 'arenaCity': 45, 'arenaState': 30}

**Sample (3 rows)**

|   gameId | gameDateTimeEst           | gameDay   | arenaCity      | arenaState   | arenaName      | gameLabel   | gameSubLabel       | gameSubtype   |   gameSequence | seriesGameNumber   | seriesText   |   weekNumber |   hometeamId |   awayteamId |
|---------:|:--------------------------|:----------|:---------------|:-------------|:---------------|:------------|:-------------------|:--------------|---------------:|:-------------------|:-------------|-------------:|-------------:|-------------:|
| 12400001 | 2024-10-04 12:00:00+00:00 | Fri       | Abu Dhabi      |              | Etihad Arena   | Preseason   | NBA Abu Dhabi Game | Global Games  |              1 |                    | Neutral Site |            0 |   1610612743 |   1610612738 |
| 12400002 | 2024-10-04 21:00:00+00:00 | Fri       | Salt Lake City | UT           | Delta Center   | Preseason   |                    |               |              2 |                    |              |            0 |   1610612762 |        15020 |
| 12400003 | 2024-10-04 22:30:00+00:00 | Fri       | Palm Desert    | CA           | Acrisure Arena | Preseason   |                    |               |              3 |                    |              |            0 |   1610612747 |   1610612750 |

## PlayerStatistics

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | firstName | TEXT | 0 | 0 |
| 1 | lastName | TEXT | 0 | 0 |
| 2 | personId | INTEGER | 0 | 0 |
| 3 | gameId | INTEGER | 0 | 0 |
| 4 | gameDate | TEXT | 0 | 0 |
| 5 | playerteamCity | TEXT | 0 | 0 |
| 6 | playerteamName | TEXT | 0 | 0 |
| 7 | opponentteamCity | TEXT | 0 | 0 |
| 8 | opponentteamName | TEXT | 0 | 0 |
| 9 | gameType | TEXT | 0 | 0 |
| 10 | gameLabel | TEXT | 0 | 0 |
| 11 | gameSubLabel | TEXT | 0 | 0 |
| 12 | seriesGameNumber | REAL | 0 | 0 |
| 13 | win | INTEGER | 0 | 0 |
| 14 | home | INTEGER | 0 | 0 |
| 15 | numMinutes | REAL | 0 | 0 |
| 16 | points | REAL | 0 | 0 |
| 17 | assists | REAL | 0 | 0 |
| 18 | blocks | REAL | 0 | 0 |
| 19 | steals | REAL | 0 | 0 |
| 20 | fieldGoalsAttempted | REAL | 0 | 0 |
| 21 | fieldGoalsMade | REAL | 0 | 0 |
| 22 | fieldGoalsPercentage | REAL | 0 | 0 |
| 23 | threePointersAttempted | REAL | 0 | 0 |
| 24 | threePointersMade | REAL | 0 | 0 |
| 25 | threePointersPercentage | REAL | 0 | 0 |
| 26 | freeThrowsAttempted | REAL | 0 | 0 |
| 27 | freeThrowsMade | REAL | 0 | 0 |
| 28 | freeThrowsPercentage | REAL | 0 | 0 |
| 29 | reboundsDefensive | REAL | 0 | 0 |
| 30 | reboundsOffensive | REAL | 0 | 0 |
| 31 | reboundsTotal | REAL | 0 | 0 |
| 32 | foulsPersonal | REAL | 0 | 0 |
| 33 | turnovers | REAL | 0 | 0 |
| 34 | plusMinusPoints | REAL | 0 | 0 |

**Profile**
- rows: 1627438
- date range: 1946-11-26 19:00:00 → 2025-06-22 20:00:00
- distinct (first 5 cols): {'firstName': 1804, 'lastName': 3142, 'personId': 5428, 'gameId': 71877, 'gameDate': 33976}

**Sample (3 rows)**

| firstName   | lastName   |   personId |   gameId | gameDate            | playerteamCity   | playerteamName   | opponentteamCity   | opponentteamName   | gameType   | gameLabel   | gameSubLabel   |   seriesGameNumber |   win |   home |   numMinutes |   points |   assists |   blocks |   steals |   fieldGoalsAttempted |   fieldGoalsMade |   fieldGoalsPercentage |   threePointersAttempted |   threePointersMade |   threePointersPercentage |   freeThrowsAttempted |   freeThrowsMade |   freeThrowsPercentage |   reboundsDefensive |   reboundsOffensive |   reboundsTotal |   foulsPersonal |   turnovers |   plusMinusPoints |
|:------------|:-----------|-----------:|---------:|:--------------------|:-----------------|:-----------------|:-------------------|:-------------------|:-----------|:------------|:---------------|-------------------:|------:|-------:|-------------:|---------:|----------:|---------:|---------:|----------------------:|-----------------:|-----------------------:|-------------------------:|--------------------:|--------------------------:|----------------------:|-----------------:|-----------------------:|--------------------:|--------------------:|----------------:|----------------:|------------:|------------------:|
| James       | Johnson    |     201949 | 42400407 | 2025-06-22 20:00:00 | Indiana          | Pacers           | Oklahoma City      | Thunder            | Playoffs   | NBA Finals  | Game 7         |                  7 |     0 |      0 |       nan    |        0 |         0 |        0 |        0 |                     0 |                0 |                  0     |                        0 |                   0 |                     0     |                     0 |                0 |                   0    |                   0 |                   0 |               0 |               0 |           0 |                 0 |
| T.J.        | McConnell  |     204456 | 42400407 | 2025-06-22 20:00:00 | Indiana          | Pacers           | Oklahoma City      | Thunder            | Playoffs   | NBA Finals  | Game 7         |                  7 |     0 |      0 |        28.05 |       16 |         3 |        1 |        1 |                    13 |                8 |                  0.615 |                        0 |                   0 |                     0     |                     0 |                0 |                   0    |                   6 |                   0 |               6 |               2 |           7 |               -12 |
| Myles       | Turner     |    1626167 | 42400407 | 2025-06-22 20:00:00 | Indiana          | Pacers           | Oklahoma City      | Thunder            | Playoffs   | NBA Finals  | Game 7         |                  7 |     0 |      0 |        23.49 |        6 |         1 |        0 |        0 |                     4 |                2 |                  0.5   |                        3 |                   1 |                     0.333 |                     4 |                1 |                   0.25 |                   3 |                   1 |               4 |               3 |           1 |               -12 |

## Players

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | personId | INTEGER | 0 | 0 |
| 1 | firstName | TEXT | 0 | 0 |
| 2 | lastName | TEXT | 0 | 0 |
| 3 | birthdate | TEXT | 0 | 0 |
| 4 | lastAttended | TEXT | 0 | 0 |
| 5 | country | TEXT | 0 | 0 |
| 6 | height | REAL | 0 | 0 |
| 7 | bodyWeight | REAL | 0 | 0 |
| 8 | guard | INTEGER | 0 | 0 |
| 9 | forward | INTEGER | 0 | 0 |
| 10 | center | INTEGER | 0 | 0 |
| 11 | draftYear | REAL | 0 | 0 |
| 12 | draftRound | REAL | 0 | 0 |
| 13 | draftNumber | REAL | 0 | 0 |

**Profile**
- rows: 6533
- date range: 1900-01-01 → 2005-12-30
- distinct (first 5 cols): {'personId': 6533, 'firstName': 2167, 'lastName': 3739, 'birthdate': 4315, 'lastAttended': 678}

**Sample (3 rows)**

|   personId | firstName   | lastName   | birthdate   | lastAttended     | country   |   height |   bodyWeight |   guard |   forward |   center |   draftYear |   draftRound |   draftNumber |
|-----------:|:------------|:-----------|:------------|:-----------------|:----------|---------:|-------------:|--------:|----------:|---------:|------------:|-------------:|--------------:|
|          2 | Byron       | Scott      | 1961-03-28  | Arizona State    | USA       |       76 |          205 |       1 |         0 |        0 |        1983 |            1 |             4 |
|          3 | Grant       | Long       | 1966-03-12  | Eastern Michigan | USA       |       81 |            3 |       0 |         0 |        0 |        1988 |            2 |            33 |
|          7 | Dan         | Schayes    | 1959-05-10  | Syracuse         | USA       |       83 |          260 |       0 |         0 |        1 |        1981 |            1 |            13 |

## TeamHistories

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | teamId | INTEGER | 0 | 0 |
| 1 | teamCity | TEXT | 0 | 0 |
| 2 | teamName | TEXT | 0 | 0 |
| 3 | teamAbbrev | TEXT | 0 | 0 |
| 4 | seasonFounded | INTEGER | 0 | 0 |
| 5 | seasonActiveTill | INTEGER | 0 | 0 |
| 6 | league | TEXT | 0 | 0 |

**Profile**
- rows: 140
- distinct (first 5 cols): {'teamId': 97, 'teamCity': 89, 'teamName': 99, 'teamAbbrev': 110, 'seasonFounded': 65}

**Sample (3 rows)**

|     teamId | teamCity   | teamName   | teamAbbrev   |   seasonFounded |   seasonActiveTill | league   |
|-----------:|:-----------|:-----------|:-------------|----------------:|-------------------:|:---------|
| 1610612737 | Tri-Cities | Blackhawks | TRI          |            1946 |               1948 | BAA      |
| 1610612737 | Milwaukee  | Hawks      | MIL          |            1951 |               1954 | NBA      |
| 1610612737 | St. Louis  | Hawks      | STL          |            1955 |               1967 | NBA      |

## TeamStatistics

| # | name | type | notnull | pk |
|---:|---|---|:---:|:--:|
| 0 | gameId | INTEGER | 0 | 0 |
| 1 | gameDate | TEXT | 0 | 0 |
| 2 | teamCity | TEXT | 0 | 0 |
| 3 | teamName | TEXT | 0 | 0 |
| 4 | teamId | INTEGER | 0 | 0 |
| 5 | opponentTeamCity | TEXT | 0 | 0 |
| 6 | opponentTeamName | TEXT | 0 | 0 |
| 7 | opponentTeamId | INTEGER | 0 | 0 |
| 8 | home | INTEGER | 0 | 0 |
| 9 | win | INTEGER | 0 | 0 |
| 10 | teamScore | INTEGER | 0 | 0 |
| 11 | opponentScore | INTEGER | 0 | 0 |
| 12 | assists | REAL | 0 | 0 |
| 13 | blocks | REAL | 0 | 0 |
| 14 | steals | REAL | 0 | 0 |
| 15 | fieldGoalsAttempted | REAL | 0 | 0 |
| 16 | fieldGoalsMade | REAL | 0 | 0 |
| 17 | fieldGoalsPercentage | REAL | 0 | 0 |
| 18 | threePointersAttempted | REAL | 0 | 0 |
| 19 | threePointersMade | REAL | 0 | 0 |
| 20 | threePointersPercentage | REAL | 0 | 0 |
| 21 | freeThrowsAttempted | REAL | 0 | 0 |
| 22 | freeThrowsMade | REAL | 0 | 0 |
| 23 | freeThrowsPercentage | REAL | 0 | 0 |
| 24 | reboundsDefensive | REAL | 0 | 0 |
| 25 | reboundsOffensive | REAL | 0 | 0 |
| 26 | reboundsTotal | REAL | 0 | 0 |
| 27 | foulsPersonal | REAL | 0 | 0 |
| 28 | turnovers | REAL | 0 | 0 |
| 29 | plusMinusPoints | REAL | 0 | 0 |
| 30 | numMinutes | REAL | 0 | 0 |
| 31 | q1Points | REAL | 0 | 0 |
| 32 | q2Points | REAL | 0 | 0 |
| 33 | q3Points | REAL | 0 | 0 |
| 34 | q4Points | REAL | 0 | 0 |
| 35 | benchPoints | REAL | 0 | 0 |
| 36 | biggestLead | REAL | 0 | 0 |
| 37 | biggestScoringRun | REAL | 0 | 0 |
| 38 | leadChanges | REAL | 0 | 0 |
| 39 | pointsFastBreak | REAL | 0 | 0 |
| 40 | pointsFromTurnovers | REAL | 0 | 0 |
| 41 | pointsInThePaint | REAL | 0 | 0 |
| 42 | pointsSecondChance | REAL | 0 | 0 |
| 43 | timesTied | REAL | 0 | 0 |
| 44 | timeoutsRemaining | REAL | 0 | 0 |
| 45 | seasonWins | REAL | 0 | 0 |
| 46 | seasonLosses | REAL | 0 | 0 |
| 47 | coachId | REAL | 0 | 0 |

**Profile**
- rows: 143758
- date range: 1946-11-26 19:00:00 → 2025-06-22 20:00:00
- distinct (first 5 cols): {'gameId': 71879, 'gameDate': 33977, 'teamCity': 46, 'teamName': 39, 'teamId': 30}

**Sample (3 rows)**

|   gameId | gameDate            | teamCity      | teamName   |     teamId | opponentTeamCity   | opponentTeamName   |   opponentTeamId |   home |   win |   teamScore |   opponentScore |   assists |   blocks |   steals |   fieldGoalsAttempted |   fieldGoalsMade |   fieldGoalsPercentage |   threePointersAttempted |   threePointersMade |   threePointersPercentage |   freeThrowsAttempted |   freeThrowsMade |   freeThrowsPercentage |   reboundsDefensive |   reboundsOffensive |   reboundsTotal |   foulsPersonal |   turnovers |   plusMinusPoints |   numMinutes |   q1Points |   q2Points |   q3Points |   q4Points |   benchPoints |   biggestLead |   biggestScoringRun |   leadChanges |   pointsFastBreak |   pointsFromTurnovers |   pointsInThePaint |   pointsSecondChance |   timesTied |   timeoutsRemaining |   seasonWins |   seasonLosses | coachId   |
|---------:|:--------------------|:--------------|:-----------|-----------:|:-------------------|:-------------------|-----------------:|-------:|------:|------------:|----------------:|----------:|---------:|---------:|----------------------:|-----------------:|-----------------------:|-------------------------:|--------------------:|--------------------------:|----------------------:|-----------------:|-----------------------:|--------------------:|--------------------:|----------------:|----------------:|------------:|------------------:|-------------:|-----------:|-----------:|-----------:|-----------:|--------------:|--------------:|--------------------:|--------------:|------------------:|----------------------:|-------------------:|---------------------:|------------:|--------------------:|-------------:|---------------:|:----------|
| 42400407 | 2025-06-22 20:00:00 | Indiana       | Pacers     | 1610612754 | Oklahoma City      | Thunder            |       1610612760 |      0 |     0 |          91 |             103 |        17 |        4 |        6 |                    70 |               29 |                  0.414 |                       28 |                  11 |                     0.393 |                    29 |               22 |                  0.759 |                  33 |                  12 |              45 |              24 |          21 |               -12 |          240 |         22 |         26 |         20 |         23 |            42 |             5 |                   7 |            11 |                 7 |                    10 |                 26 |                   15 |          10 |                   0 |            3 |              4 |           |
| 42400407 | 2025-06-22 20:00:00 | Oklahoma City | Thunder    | 1610612760 | Indiana            | Pacers             |       1610612754 |      1 |     1 |         103 |              91 |        20 |        8 |       14 |                    87 |               35 |                  0.402 |                       40 |                  11 |                     0.275 |                    31 |               22 |                  0.71  |                  27 |                  13 |              40 |              23 |           7 |                12 |          240 |         25 |         22 |         34 |         22 |            20 |            22 |                  13 |            11 |                 9 |                    32 |                 40 |                   22 |          10 |                   0 |            4 |              3 |           |
| 42400406 | 2025-06-19 20:30:00 | Indiana       | Pacers     | 1610612754 | Oklahoma City      | Thunder            |       1610612760 |      1 |     1 |         108 |              91 |        23 |        5 |       16 |                    92 |               38 |                  0.413 |                       42 |                  15 |                     0.357 |                    25 |               17 |                  0.68  |                  35 |                  11 |              46 |              17 |          10 |                17 |          240 |         28 |         36 |         26 |         18 |            48 |            31 |                  12 |             1 |                22 |                    19 |                 30 |                   14 |           1 |                   1 |            3 |              3 |           |
