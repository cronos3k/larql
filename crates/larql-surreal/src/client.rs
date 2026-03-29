//! SurrealDB HTTP client wrapper.

use crate::error::SurrealError;

/// HTTP client wrapper for SurrealDB's /sql endpoint.
pub struct SurrealClient {
    client: reqwest::blocking::Client,
    url: String,
    ns: String,
    db: String,
}

impl SurrealClient {
    pub fn new(url: &str, ns: &str, db: &str, user: &str, pass: &str) -> Self {
        let url = url.trim_end_matches('/').to_string();

        let client = reqwest::blocking::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .default_headers({
                let mut headers = reqwest::header::HeaderMap::new();
                let auth = base64_encode(&format!("{user}:{pass}"));
                headers.insert(
                    reqwest::header::AUTHORIZATION,
                    format!("Basic {auth}").parse().unwrap(),
                );
                headers
            })
            .build()
            .expect("failed to build HTTP client");

        Self {
            client,
            url,
            ns: ns.to_string(),
            db: db.to_string(),
        }
    }

    /// Execute a SurQL query and return the raw JSON response.
    pub fn query(&self, sql: &str) -> Result<serde_json::Value, SurrealError> {
        let resp = self
            .client
            .post(format!("{}/sql", self.url))
            .header("surreal-ns", &self.ns)
            .header("surreal-db", &self.db)
            .header("Accept", "application/json")
            .body(sql.to_string())
            .send()
            .map_err(|e| SurrealError::Surreal(format!("HTTP error: {e}")))?;

        let status = resp.status();
        let body = resp
            .text()
            .map_err(|e| SurrealError::Surreal(format!("read error: {e}")))?;

        if !status.is_success() {
            return Err(SurrealError::Surreal(format!("HTTP {status}: {body}")));
        }

        serde_json::from_str(&body)
            .map_err(|e| SurrealError::Surreal(format!("JSON parse: {e}: {body}")))
    }

    /// Execute a SurQL statement, ignoring the response.
    pub fn exec(&self, sql: &str) -> Result<(), SurrealError> {
        self.query(sql)?;
        Ok(())
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn ns(&self) -> &str {
        &self.ns
    }

    pub fn db(&self) -> &str {
        &self.db
    }
}

pub fn base64_encode(input: &str) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let bytes = input.as_bytes();
    let mut result = String::with_capacity(bytes.len().div_ceil(3) * 4);

    for chunk in bytes.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;

        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}
