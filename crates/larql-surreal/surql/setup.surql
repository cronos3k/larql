-- Setup namespace and database
-- Parameters: $ns, $db (substituted by the loader)
DEFINE NAMESPACE IF NOT EXISTS {ns};
USE NS {ns};
DEFINE DATABASE IF NOT EXISTS {db};
USE DB {db};
