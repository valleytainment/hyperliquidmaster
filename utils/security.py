"""
Security management for the trading bot
Handles private key encryption, storage, and authentication
"""

import os
import json
import base64
import getpass
from pathlib import Path
from typing import Optional, Dict, Any
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import keyring
from utils.logger import get_logger

logger = get_logger(__name__)


class SecurityManager:
    """Handles secure storage and retrieval of sensitive data"""
    
    def __init__(self, app_name: str = "HyperliquidTradingBot"):
        """
        Initialize security manager
        
        Args:
            app_name: Application name for keyring storage
        """
        self.app_name = app_name
        self.secrets_dir = Path("secrets")
        self.secrets_dir.mkdir(exist_ok=True)
        
        # Encrypted storage file
        self.encrypted_file = self.secrets_dir / "encrypted_keys.dat"
        
        logger.info("Security manager initialized")
        
    def clear_private_key(self):
        """
        Clear stored private key from keyring and local storage
        
        Returns:
            bool: True if successful
        """
        try:
            # Delete from keyring
            try:
                keyring.delete_password(self.app_name, "private_key")
            except Exception as e:
                logger.warning(f"No credentials found in keyring for private_key: {e}")
            
            # Delete encrypted file if exists
            if self.encrypted_file.exists():
                self.secure_delete_file(self.encrypted_file)
                
            logger.info("Private key cleared successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear private key: {e}")
            return False
    
    def _derive_key(self, password: str, salt: bytes) -> bytes:
        """Derive encryption key from password"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        return base64.urlsafe_b64encode(kdf.derive(password.encode()))
    
    def encrypt_private_key(self, private_key: str, password: str = None) -> bool:
        """
        Encrypt and store private key
        
        Args:
            private_key: The private key to encrypt
            password: Password for encryption (will prompt if not provided)
            
        Returns:
            bool: True if successful
        """
        try:
            if not password:
                password = getpass.getpass("Enter password to encrypt private key: ")
                confirm_password = getpass.getpass("Confirm password: ")
                
                if password != confirm_password:
                    logger.error("Passwords do not match")
                    return False
            
            # Generate salt
            salt = os.urandom(16)
            
            # Derive key
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            # Encrypt private key
            encrypted_key = fernet.encrypt(private_key.encode())
            
            # Store encrypted data
            encrypted_data = {
                'salt': base64.b64encode(salt).decode(),
                'encrypted_key': base64.b64encode(encrypted_key).decode(),
                'created_at': str(Path().cwd())  # Timestamp
            }
            
            with open(self.encrypted_file, 'w') as f:
                json.dump(encrypted_data, f)
            
            # Set file permissions (Unix only)
            if os.name != 'nt':
                os.chmod(self.encrypted_file, 0o600)
            
            logger.info("Private key encrypted and stored successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to encrypt private key: {e}")
            return False
    
    def decrypt_private_key(self, password: str = None) -> Optional[str]:
        """
        Decrypt and retrieve private key
        
        Args:
            password: Password for decryption (will prompt if not provided)
            
        Returns:
            str: Decrypted private key or None if failed
        """
        try:
            if not self.encrypted_file.exists():
                logger.error("No encrypted private key found")
                return None
            
            if not password:
                password = getpass.getpass("Enter password to decrypt private key: ")
            
            # Load encrypted data
            with open(self.encrypted_file, 'r') as f:
                encrypted_data = json.load(f)
            
            # Extract components
            salt = base64.b64decode(encrypted_data['salt'])
            encrypted_key = base64.b64decode(encrypted_data['encrypted_key'])
            
            # Derive key
            key = self._derive_key(password, salt)
            fernet = Fernet(key)
            
            # Decrypt private key
            private_key = fernet.decrypt(encrypted_key).decode()
            
            logger.info("Private key decrypted successfully")
            return private_key
            
        except Exception as e:
            logger.error(f"Failed to decrypt private key: {e}")
            return None
    
    def store_in_keyring(self, username: str, password: str) -> bool:
        """
        Store credentials in system keyring
        
        Args:
            username: Username/identifier
            password: Password/private key to store
            
        Returns:
            bool: True if successful
        """
        try:
            keyring.set_password(self.app_name, username, password)
            logger.info(f"Credentials stored in keyring for {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store in keyring: {e}")
            return False
    
    def get_from_keyring(self, username: str) -> Optional[str]:
        """
        Retrieve credentials from system keyring
        
        Args:
            username: Username/identifier
            
        Returns:
            str: Retrieved password/private key or None if not found
        """
        try:
            password = keyring.get_password(self.app_name, username)
            if password:
                logger.info(f"Credentials retrieved from keyring for {username}")
            else:
                logger.warning(f"No credentials found in keyring for {username}")
            return password
            
        except Exception as e:
            logger.error(f"Failed to retrieve from keyring: {e}")
            return None
    
    def delete_from_keyring(self, username: str) -> bool:
        """
        Delete credentials from system keyring
        
        Args:
            username: Username/identifier
            
        Returns:
            bool: True if successful
        """
        try:
            keyring.delete_password(self.app_name, username)
            logger.info(f"Credentials deleted from keyring for {username}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete from keyring: {e}")
            return False
    
    def get_private_key(self, method: str = "auto") -> Optional[str]:
        """
        Get private key using the specified method
        
        Args:
            method: Method to use ("auto", "file", "keyring", "env", "prompt")
            
        Returns:
            str: Private key or None if not found
        """
        if method == "auto":
            # Try methods in order of preference
            methods = ["env", "keyring", "file", "prompt"]
        else:
            methods = [method]
        
        for m in methods:
            try:
                if m == "env":
                    # Try environment variable
                    private_key = os.getenv("HYPERLIQUID_PRIVATE_KEY")
                    if private_key:
                        logger.info("Private key loaded from environment variable")
                        return private_key
                
                elif m == "keyring":
                    # Try system keyring
                    private_key = self.get_from_keyring("private_key")
                    if private_key:
                        return private_key
                
                elif m == "file":
                    # Try encrypted file
                    private_key = self.decrypt_private_key()
                    if private_key:
                        return private_key
                
                elif m == "prompt":
                    # Prompt user
                    private_key = getpass.getpass("Enter your private key: ")
                    if private_key:
                        logger.info("Private key entered manually")
                        return private_key
                
            except Exception as e:
                logger.debug(f"Method {m} failed: {e}")
                continue
        
        logger.error("Failed to retrieve private key using any method")
        return None
    
    def setup_private_key(self) -> bool:
        """
        Interactive setup for private key storage
        
        Returns:
            bool: True if setup successful
        """
        print("\n=== Private Key Setup ===")
        print("Choose how to store your private key:")
        print("1. Encrypted file (recommended)")
        print("2. System keyring")
        print("3. Environment variable (less secure)")
        print("4. Skip setup")
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            private_key = getpass.getpass("Enter your private key: ")
            return self.encrypt_private_key(private_key)
        
        elif choice == "2":
            private_key = getpass.getpass("Enter your private key: ")
            return self.store_in_keyring("private_key", private_key)
        
        elif choice == "3":
            print("Set the environment variable HYPERLIQUID_PRIVATE_KEY")
            print("Example: export HYPERLIQUID_PRIVATE_KEY='your_private_key_here'")
            return True
        
        elif choice == "4":
            print("Setup skipped. You'll need to provide the private key manually.")
            return True
        
        else:
            print("Invalid choice")
            return False
    
    def validate_private_key(self, private_key: str) -> bool:
        """
        Validate private key format
        
        Args:
            private_key: Private key to validate
            
        Returns:
            bool: True if valid format
        """
        try:
            # Basic validation - should be 64 hex characters (32 bytes)
            if len(private_key) == 64:
                int(private_key, 16)  # Check if valid hex
                return True
            
            # Check if it starts with 0x
            if private_key.startswith('0x') and len(private_key) == 66:
                int(private_key[2:], 16)  # Check if valid hex
                return True
            
            return False
            
        except ValueError:
            return False
    
    def generate_api_key(self) -> str:
        """Generate a random API key for internal use"""
        return base64.urlsafe_b64encode(os.urandom(32)).decode()
    
    def hash_password(self, password: str, salt: bytes = None) -> tuple:
        """
        Hash a password with salt
        
        Args:
            password: Password to hash
            salt: Salt bytes (will generate if not provided)
            
        Returns:
            tuple: (hashed_password, salt)
        """
        if salt is None:
            salt = os.urandom(32)
        
        pwdhash = hashes.Hash(hashes.SHA256())
        pwdhash.update(salt + password.encode())
        
        return pwdhash.finalize(), salt
    
    def verify_password(self, password: str, hashed_password: bytes, salt: bytes) -> bool:
        """
        Verify a password against its hash
        
        Args:
            password: Password to verify
            hashed_password: Stored hash
            salt: Salt used for hashing
            
        Returns:
            bool: True if password matches
        """
        try:
            new_hash, _ = self.hash_password(password, salt)
            return new_hash == hashed_password
        except Exception:
            return False
    
    def secure_delete_file(self, file_path: Path) -> bool:
        """
        Securely delete a file by overwriting it
        
        Args:
            file_path: Path to file to delete
            
        Returns:
            bool: True if successful
        """
        try:
            if not file_path.exists():
                return True
            
            # Get file size
            file_size = file_path.stat().st_size
            
            # Overwrite with random data multiple times
            with open(file_path, 'r+b') as f:
                for _ in range(3):
                    f.seek(0)
                    f.write(os.urandom(file_size))
                    f.flush()
                    os.fsync(f.fileno())
            
            # Delete the file
            file_path.unlink()
            
            logger.info(f"File securely deleted: {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to securely delete file: {e}")
            return False
    
    def backup_encrypted_keys(self, backup_path: str) -> bool:
        """
        Create a backup of encrypted keys
        
        Args:
            backup_path: Path for backup file
            
        Returns:
            bool: True if successful
        """
        try:
            if not self.encrypted_file.exists():
                logger.error("No encrypted keys file to backup")
                return False
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy encrypted file
            import shutil
            shutil.copy2(self.encrypted_file, backup_path)
            
            logger.info(f"Encrypted keys backed up to {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to backup encrypted keys: {e}")
            return False
    
    def restore_encrypted_keys(self, backup_path: str) -> bool:
        """
        Restore encrypted keys from backup
        
        Args:
            backup_path: Path to backup file
            
        Returns:
            bool: True if successful
        """
        try:
            backup_path = Path(backup_path)
            
            if not backup_path.exists():
                logger.error(f"Backup file not found: {backup_path}")
                return False
            
            # Copy backup to encrypted file location
            import shutil
            shutil.copy2(backup_path, self.encrypted_file)
            
            # Set proper permissions
            if os.name != 'nt':
                os.chmod(self.encrypted_file, 0o600)
            
            logger.info(f"Encrypted keys restored from {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore encrypted keys: {e}")
            return False
    
    def clear_all_credentials(self) -> bool:
        """
        Clear all stored credentials (use with caution)
        
        Returns:
            bool: True if successful
        """
        try:
            success = True
            
            # Delete encrypted file
            if self.encrypted_file.exists():
                success &= self.secure_delete_file(self.encrypted_file)
            
            # Delete from keyring
            try:
                self.delete_from_keyring("private_key")
            except Exception:
                pass  # May not exist
            
            # Clear environment variable (for current session)
            if "HYPERLIQUID_PRIVATE_KEY" in os.environ:
                del os.environ["HYPERLIQUID_PRIVATE_KEY"]
            
            logger.warning("All credentials cleared")
            return success
            
        except Exception as e:
            logger.error(f"Failed to clear credentials: {e}")
            return False

