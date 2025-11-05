"""
File Validator
Validates uploaded files for size, format, and security before processing.
"""
#src/data_processing/file_validator.py
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import python-magic for MIME type detection (optional)
try:
    import magic
    HAS_MAGIC = True
except ImportError:
    HAS_MAGIC = False

logger = logging.getLogger(__name__)
if not HAS_MAGIC:
    logger.warning("python-magic not installed. File type validation will use extensions only.")


class FileValidationError(Exception):
    """Raised when file validation fails."""
    pass


class FileValidator:
    """Validates uploaded files before processing."""

    # Supported file extensions and their MIME types
    SUPPORTED_FORMATS = {
        # Excel formats
        '.xlsb': ['application/vnd.ms-excel.sheet.binary.macroEnabled.12'],
        '.xlsx': ['application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'],
        '.xls': ['application/vnd.ms-excel'],
        # CSV formats
        '.csv': ['text/csv', 'text/plain', 'application/csv'],
        '.tsv': ['text/tab-separated-values', 'text/plain'],
        # Parquet
        '.parquet': ['application/octet-stream', 'application/parquet'],
    }

    # Quota limits
    MAX_FILE_SIZE_BYTES = 200 * 1024 * 1024  # 200MB per file
    MAX_FILES_PER_CHAT = 5
    MAX_USER_STORAGE_BYTES = 500 * 1024 * 1024  # 500MB per user

    # Security patterns
    DANGEROUS_EXTENSIONS = [
        '.exe', '.dll', '.bat', '.cmd', '.sh', '.ps1', '.vbs',
        '.js', '.jar', '.app', '.dmg', '.pkg', '.deb', '.rpm'
    ]

    def __init__(self, max_file_size: Optional[int] = None):
        """
        Initialize file validator.

        Args:
            max_file_size: Override default max file size in bytes
        """
        self.max_file_size = max_file_size or self.MAX_FILE_SIZE_BYTES

    def validate_file(
        self,
        file_path: str,
        original_filename: str,
        current_chat_file_count: int = 0,
        current_user_storage: int = 0
    ) -> Dict[str, any]:
        """
        Validate uploaded file.

        Args:
            file_path: Path to uploaded file
            original_filename: Original filename from upload
            current_chat_file_count: Number of files already in chat
            current_user_storage: Total storage used by user in bytes

        Returns:
            Dictionary with validation result

        Raises:
            FileValidationError: If validation fails
        """
        file_path = Path(file_path)

        # Check file exists
        if not file_path.exists():
            raise FileValidationError(f"File not found: {file_path}")

        # Get file size
        file_size = file_path.stat().st_size

        # Validate file size
        if file_size == 0:
            raise FileValidationError("File is empty")

        if file_size > self.max_file_size:
            raise FileValidationError(
                f"File too large: {file_size / 1024 / 1024:.2f}MB. "
                f"Maximum allowed: {self.max_file_size / 1024 / 1024:.2f}MB"
            )

        # Validate chat file count
        if current_chat_file_count >= self.MAX_FILES_PER_CHAT:
            raise FileValidationError(
                f"Maximum files per chat ({self.MAX_FILES_PER_CHAT}) exceeded"
            )

        # Validate user storage quota
        if (current_user_storage + file_size) > self.MAX_USER_STORAGE_BYTES:
            raise FileValidationError(
                f"User storage quota exceeded. "
                f"Current: {current_user_storage / 1024 / 1024:.2f}MB, "
                f"File: {file_size / 1024 / 1024:.2f}MB, "
                f"Limit: {self.MAX_USER_STORAGE_BYTES / 1024 / 1024:.2f}MB"
            )

        # Validate filename
        sanitized_name = self.sanitize_filename(original_filename)

        # Validate file extension
        extension = sanitized_name.suffix.lower()

        if extension not in self.SUPPORTED_FORMATS:
            supported = ', '.join(self.SUPPORTED_FORMATS.keys())
            raise FileValidationError(
                f"Unsupported file format: {extension}. "
                f"Supported formats: {supported}"
            )

        # Security check: dangerous extensions
        if extension in self.DANGEROUS_EXTENSIONS:
            raise FileValidationError(
                f"File type not allowed for security reasons: {extension}"
            )

        # Validate MIME type (optional but recommended)
        mime_type = None
        if HAS_MAGIC:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
                allowed_mimes = self.SUPPORTED_FORMATS[extension]

                if mime_type not in allowed_mimes:
                    logger.warning(
                        f"MIME type mismatch: expected {allowed_mimes}, got {mime_type}"
                    )
                    # Don't fail on MIME mismatch, just log warning

            except Exception as e:
                logger.warning(f"Could not detect MIME type: {e}")

        logger.info(
            f"File validation passed: {sanitized_name} "
            f"({file_size / 1024:.2f} KB, {extension})"
        )

        return {
            "valid": True,
            "sanitized_filename": sanitized_name.name,
            "extension": extension,
            "size_bytes": file_size,
            "mime_type": mime_type
        }

    def sanitize_filename(self, filename: str) -> Path:
        """
        Sanitize filename for safe storage.

        Args:
            filename: Original filename

        Returns:
            Path object with sanitized filename
        """
        # Get name and extension
        path = Path(filename)
        name = path.stem
        ext = path.suffix.lower()

        # Remove or replace dangerous characters
        # Allow: letters, numbers, underscore, hyphen, dot, space
        name = re.sub(r'[^\w\s\-\.]', '_', name)

        # Replace multiple spaces/underscores with single
        name = re.sub(r'[\s_]+', '_', name)

        # Remove leading/trailing underscores and dots
        name = name.strip('_.')

        # Limit length
        if len(name) > 100:
            name = name[:100]

        # Ensure not empty
        if not name:
            name = "upload"

        return Path(f"{name}{ext}")

    def validate_sheet_names(self, sheet_names: List[str]) -> Tuple[bool, Optional[str]]:
        """
        Validate Excel sheet names for SQL compatibility.

        Args:
            sheet_names: List of sheet names from Excel file

        Returns:
            Tuple of (valid, error_message)
        """
        if not sheet_names:
            return False, "No sheets found in file"

        sql_keywords = {
            'SELECT', 'FROM', 'WHERE', 'INSERT', 'UPDATE', 'DELETE',
            'CREATE', 'DROP', 'ALTER', 'TABLE', 'INDEX', 'VIEW'
        }

        for sheet_name in sheet_names:
            # Check for empty names
            if not sheet_name or sheet_name.strip() == '':
                return False, "Sheet has empty name"

            # Check for SQL keywords
            if sheet_name.upper() in sql_keywords:
                return False, f"Sheet name '{sheet_name}' is a SQL keyword"

            # Check for problematic characters
            if not re.match(r'^[a-zA-Z0-9_\s\-]+$', sheet_name):
                return False, f"Sheet name '{sheet_name}' contains invalid characters"

        return True, None

    def get_quota_info(
        self,
        current_file_count: int,
        current_storage_bytes: int
    ) -> Dict[str, any]:
        """
        Get quota usage information.

        Args:
            current_file_count: Number of files in chat
            current_storage_bytes: Total storage used in bytes

        Returns:
            Dictionary with quota information
        """
        return {
            "files": {
                "current": current_file_count,
                "limit": self.MAX_FILES_PER_CHAT,
                "remaining": max(0, self.MAX_FILES_PER_CHAT - current_file_count),
                "percentage": (current_file_count / self.MAX_FILES_PER_CHAT) * 100
            },
            "storage": {
                "current_bytes": current_storage_bytes,
                "current_mb": current_storage_bytes / 1024 / 1024,
                "limit_bytes": self.MAX_USER_STORAGE_BYTES,
                "limit_mb": self.MAX_USER_STORAGE_BYTES / 1024 / 1024,
                "remaining_bytes": max(0, self.MAX_USER_STORAGE_BYTES - current_storage_bytes),
                "remaining_mb": max(0, (self.MAX_USER_STORAGE_BYTES - current_storage_bytes) / 1024 / 1024),
                "percentage": (current_storage_bytes / self.MAX_USER_STORAGE_BYTES) * 100
            }
        }


# Example CLI entry point removed; this module is used by the FastAPI app.
