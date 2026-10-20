import unittest
from unittest.mock import Mock, patch

from active.secret_store import (
    GEMINI_USERNAME,
    SERVICE_NAME,
    get_stored_gemini_api_key,
    store_gemini_api_key,
)
from active.store_gemini_api_key import main as store_gemini_api_key_main


class GeminiSecretStoreTests(unittest.TestCase):
    def test_setup_command_prompts_twice_and_saves_matching_key(self):
        with (
            patch(
                "active.store_gemini_api_key.getpass",
                side_effect=["entered-key", "entered-key"],
            ),
            patch("active.store_gemini_api_key.store_gemini_api_key") as store_key,
        ):
            store_gemini_api_key_main()

        store_key.assert_called_once_with("entered-key")

    def test_setup_command_does_not_save_mismatched_key(self):
        with (
            patch(
                "active.store_gemini_api_key.getpass",
                side_effect=["entered-key", "different-key"],
            ),
            patch("active.store_gemini_api_key.store_gemini_api_key") as store_key,
        ):
            with self.assertRaisesRegex(SystemExit, "did not match"):
                store_gemini_api_key_main()

        store_key.assert_not_called()

    def test_reads_key_from_windows_credential_backend(self):
        keyring = Mock()
        keyring.get_password.return_value = "stored-key"
        with patch("active.secret_store._create_windows_keyring", return_value=keyring):
            result = get_stored_gemini_api_key()

        self.assertEqual(result, "stored-key")
        keyring.get_password.assert_called_once_with(SERVICE_NAME, GEMINI_USERNAME)

    def test_stores_trimmed_key_in_windows_credential_backend(self):
        keyring = Mock()
        with patch("active.secret_store._create_windows_keyring", return_value=keyring):
            store_gemini_api_key("  saved-key  ")

        keyring.set_password.assert_called_once_with(
            SERVICE_NAME,
            GEMINI_USERNAME,
            "saved-key",
        )

    def test_rejects_blank_key_without_accessing_credential_backend(self):
        with patch("active.secret_store._create_windows_keyring") as keyring_factory:
            with self.assertRaisesRegex(ValueError, "non-empty"):
                store_gemini_api_key("  ")

        keyring_factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
