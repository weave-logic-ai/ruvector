//! ML-DSA-65 placeholder and dual (Ed25519 + ML-DSA-65) segment signing.
//!
//! The ML-DSA-65 implementation uses HMAC-SHA3-256 as a deterministic
//! placeholder. Enable the `production-ml-dsa` feature (future) for
//! real Dilithium3 signatures.

use alloc::vec::Vec;

use rvf_types::{SegmentHeader, SignatureFooter};

#[cfg(feature = "ed25519")]
use ed25519_dalek::SigningKey;

/// ML-DSA-65 algorithm identifier (matches `SignatureAlgo::MlDsa65`).
const SIG_ALGO_ML_DSA_65: u16 = 1;

/// ML-DSA-65 signature length (FIPS 204).
pub const ML_DSA_65_SIG_LEN: u16 = 3309;

/// ML-DSA-65 secret key (HMAC key in placeholder mode).
#[derive(Clone)]
pub struct MlDsa65Key {
    /// 32-byte key used for HMAC-based placeholder signing.
    key: [u8; 32],
}

/// ML-DSA-65 public key (verification key).
#[derive(Clone)]
pub struct MlDsa65VerifyKey {
    /// 32-byte key used for HMAC-based placeholder verification.
    key: [u8; 32],
}

impl MlDsa65Key {
    /// Create a key from raw bytes.
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self { key: bytes }
    }

    /// Generate a keypair (in placeholder mode, signing key = verify key).
    pub fn generate(seed: &[u8]) -> (Self, MlDsa65VerifyKey) {
        use crate::hash::shake256_256;
        let key = shake256_256(seed);
        (Self { key }, MlDsa65VerifyKey { key })
    }

    /// Get the corresponding verification key.
    pub fn verifying_key(&self) -> MlDsa65VerifyKey {
        MlDsa65VerifyKey { key: self.key }
    }
}

/// Dual signing key: Ed25519 + ML-DSA-65.
#[cfg(feature = "ed25519")]
pub struct DualKey {
    pub ed25519: SigningKey,
    pub ml_dsa: MlDsa65Key,
}

/// Dual verification key.
#[cfg(feature = "ed25519")]
pub struct DualVerifyKey {
    pub ed25519: ed25519_dalek::VerifyingKey,
    pub ml_dsa: MlDsa65VerifyKey,
}

/// Sign a segment with ML-DSA-65 (placeholder), producing a `SignatureFooter`.
pub fn sign_segment_ml_dsa(
    header: &SegmentHeader,
    payload: &[u8],
    key: &MlDsa65Key,
) -> SignatureFooter {
    let msg = super::sign::build_signed_data_pub(header, payload);
    let sig = hmac_sign(&key.key, &msg);

    let mut signature = [0u8; SignatureFooter::MAX_SIG_LEN];
    let len = sig.len().min(ML_DSA_65_SIG_LEN as usize);
    signature[..len].copy_from_slice(&sig[..len]);

    SignatureFooter {
        sig_algo: SIG_ALGO_ML_DSA_65,
        sig_length: ML_DSA_65_SIG_LEN,
        signature,
        footer_length: SignatureFooter::compute_footer_length(ML_DSA_65_SIG_LEN),
    }
}

/// Verify an ML-DSA-65 (placeholder) segment signature.
pub fn verify_segment_ml_dsa(
    header: &SegmentHeader,
    payload: &[u8],
    footer: &SignatureFooter,
    pubkey: &MlDsa65VerifyKey,
) -> bool {
    if footer.sig_algo != SIG_ALGO_ML_DSA_65 {
        return false;
    }
    let msg = super::sign::build_signed_data_pub(header, payload);
    let expected = hmac_sign(&pubkey.key, &msg);
    let len = expected.len().min(ML_DSA_65_SIG_LEN as usize);
    footer.signature[..len] == expected[..len]
}

/// Sign a segment with both Ed25519 and ML-DSA-65, returning both footers.
#[cfg(feature = "ed25519")]
pub fn dual_sign_segment(
    header: &SegmentHeader,
    payload: &[u8],
    key: &DualKey,
) -> (SignatureFooter, SignatureFooter) {
    let ed_footer = super::sign::sign_segment(header, payload, &key.ed25519);
    let ml_footer = sign_segment_ml_dsa(header, payload, &key.ml_dsa);
    (ed_footer, ml_footer)
}

/// Verify a dual-signed segment (both Ed25519 and ML-DSA-65 must verify).
#[cfg(feature = "ed25519")]
pub fn verify_dual_segment(
    header: &SegmentHeader,
    payload: &[u8],
    ed_footer: &SignatureFooter,
    ml_footer: &SignatureFooter,
    keys: &DualVerifyKey,
) -> bool {
    super::sign::verify_segment(header, payload, ed_footer, &keys.ed25519)
        && verify_segment_ml_dsa(header, payload, ml_footer, &keys.ml_dsa)
}

/// HMAC-SHA3-256 placeholder for ML-DSA-65.
///
/// Produces a deterministic 3309-byte "signature" by repeatedly hashing
/// with the key. This is NOT quantum-resistant -- it is a placeholder
/// that validates the wire format and dual-signing flow.
fn hmac_sign(key: &[u8; 32], message: &[u8]) -> Vec<u8> {
    use crate::hash::shake256_256;

    // Build HMAC-like construction: H(key || message || key)
    let mut input = Vec::with_capacity(32 + message.len() + 32);
    input.extend_from_slice(key);
    input.extend_from_slice(message);
    input.extend_from_slice(key);

    // Generate 3309 bytes by repeatedly hashing
    let mut sig = Vec::with_capacity(ML_DSA_65_SIG_LEN as usize);
    let mut block = shake256_256(&input);
    while sig.len() < ML_DSA_65_SIG_LEN as usize {
        sig.extend_from_slice(&block);
        // Chain: next block = H(prev_block || key)
        let mut next_input = Vec::with_capacity(64);
        next_input.extend_from_slice(&block);
        next_input.extend_from_slice(key);
        block = shake256_256(&next_input);
    }
    sig.truncate(ML_DSA_65_SIG_LEN as usize);
    sig
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_header() -> SegmentHeader {
        let mut h = SegmentHeader::new(0x01, 42);
        h.timestamp_ns = 1_000_000_000;
        h.payload_length = 100;
        h
    }

    #[test]
    fn ml_dsa_sign_verify() {
        let (key, vk) = MlDsa65Key::generate(b"test seed");
        let header = make_test_header();
        let payload = b"test payload";

        let footer = sign_segment_ml_dsa(&header, payload, &key);
        assert_eq!(footer.sig_algo, SIG_ALGO_ML_DSA_65);
        assert_eq!(footer.sig_length, ML_DSA_65_SIG_LEN);

        assert!(verify_segment_ml_dsa(&header, payload, &footer, &vk));
    }

    #[test]
    fn ml_dsa_wrong_key_fails() {
        let (key, _) = MlDsa65Key::generate(b"key1");
        let (_, wrong_vk) = MlDsa65Key::generate(b"key2");
        let header = make_test_header();
        let payload = b"test";

        let footer = sign_segment_ml_dsa(&header, payload, &key);
        assert!(!verify_segment_ml_dsa(&header, payload, &footer, &wrong_vk));
    }

    #[test]
    fn ml_dsa_tampered_payload_fails() {
        let (key, vk) = MlDsa65Key::generate(b"test");
        let header = make_test_header();

        let footer = sign_segment_ml_dsa(&header, b"original", &key);
        assert!(!verify_segment_ml_dsa(&header, b"tampered", &footer, &vk));
    }

    #[test]
    #[cfg(feature = "ed25519")]
    fn dual_sign_verify() {
        use ed25519_dalek::SigningKey;
        use rand::rngs::OsRng;

        let ed_key = SigningKey::generate(&mut OsRng);
        let (ml_key, ml_vk) = MlDsa65Key::generate(b"dual test");

        let dual = DualKey {
            ed25519: ed_key.clone(),
            ml_dsa: ml_key,
        };
        let dual_vk = DualVerifyKey {
            ed25519: ed_key.verifying_key(),
            ml_dsa: ml_vk,
        };

        let header = make_test_header();
        let payload = b"dual signing test";

        let (ed_footer, ml_footer) = dual_sign_segment(&header, payload, &dual);
        assert!(verify_dual_segment(
            &header, payload, &ed_footer, &ml_footer, &dual_vk
        ));
    }

    #[test]
    fn footer_size_correct() {
        assert_eq!(
            SignatureFooter::compute_footer_length(ML_DSA_65_SIG_LEN),
            3317 // 2 + 2 + 3309 + 4
        );
    }

    #[test]
    fn hmac_deterministic() {
        let key = [42u8; 32];
        let msg = b"hello";
        let sig1 = hmac_sign(&key, msg);
        let sig2 = hmac_sign(&key, msg);
        assert_eq!(sig1, sig2);
        assert_eq!(sig1.len(), ML_DSA_65_SIG_LEN as usize);
    }
}
