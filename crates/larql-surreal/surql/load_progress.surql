-- Load progress tracking table
-- Records which layers have been loaded per component table

DEFINE TABLE IF NOT EXISTS load_progress SCHEMAFULL;

DEFINE FIELD IF NOT EXISTS table_name     ON load_progress TYPE string;
DEFINE FIELD IF NOT EXISTS layer          ON load_progress TYPE int;
DEFINE FIELD IF NOT EXISTS vectors_loaded ON load_progress TYPE int;
DEFINE FIELD IF NOT EXISTS completed      ON load_progress TYPE bool;
DEFINE FIELD IF NOT EXISTS timestamp      ON load_progress TYPE datetime;
