use larql_surreal::base64_encode;

#[test]
fn base64_encode_empty_string() {
    assert_eq!(base64_encode(""), "");
}

#[test]
fn base64_encode_single_char() {
    // 'a' -> base64 "YQ=="
    assert_eq!(base64_encode("a"), "YQ==");
}

#[test]
fn base64_encode_two_chars() {
    // 'ab' -> base64 "YWI="
    assert_eq!(base64_encode("ab"), "YWI=");
}

#[test]
fn base64_encode_three_chars() {
    // 'abc' -> base64 "YWJj"
    assert_eq!(base64_encode("abc"), "YWJj");
}

#[test]
fn base64_encode_user_pass() {
    // "root:root" -> base64 "cm9vdDpyb290"
    assert_eq!(base64_encode("root:root"), "cm9vdDpyb290");
}

#[test]
fn base64_encode_longer_string() {
    // "hello world" -> base64 "aGVsbG8gd29ybGQ="
    assert_eq!(base64_encode("hello world"), "aGVsbG8gd29ybGQ=");
}
