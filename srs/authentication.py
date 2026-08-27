"""Authentication used for calls from the SRS website to this API."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from django.conf import settings
from django.core.exceptions import ValidationError
from django.core.validators import validate_email
from drf_spectacular.extensions import OpenApiAuthenticationExtension
from rest_framework.authentication import BaseAuthentication
from rest_framework.exceptions import AuthenticationFailed


API_KEY_HEADER = "X-SRS-API-Key"
USER_ID_HEADER = "X-SRS-User-ID"
USER_EMAIL_HEADER = "X-SRS-User-Email"


@dataclass(frozen=True)
class SRSServiceCaller:
    """Authenticated SRS website caller with optional audit details."""

    user_id: int | None = None
    email: str = ""

    # The shared key authenticates the caller; these fields satisfy DRF.
    is_authenticated: bool = True
    is_active: bool = True
    is_staff: bool = False

    @property
    def id(self):
        return self.user_id

    @property
    def pk(self):
        return self.user_id

    def __str__(self):
        return self.email or (
            f"SRS website user {self.user_id}"
            if self.user_id is not None
            else "SRS website"
        )


class SRSSharedKeyAuthentication(BaseAuthentication):
    """Authenticate calls using the key shared with the SRS website."""

    def authenticate(self, request):
        expected_key = getattr(settings, "SRS_API_SHARED_KEY", "")
        supplied_key = request.headers.get(API_KEY_HEADER, "")

        # An unset key must never make the API public by accident.
        if not expected_key:
            raise AuthenticationFailed("API authentication is not configured.")

        if not supplied_key or not secrets.compare_digest(
            supplied_key.encode("utf-8"),
            expected_key.encode("utf-8"),
        ):
            raise AuthenticationFailed("Invalid or missing API key.")

        return self._build_caller(request), None

    def authenticate_header(self, request):
        # A challenge makes failed authentication return 401 instead of 403.
        return 'ApiKey realm="srs-api"'

    def _build_caller(self, request):
        raw_user_id = request.headers.get(USER_ID_HEADER, "").strip()
        raw_email = request.headers.get(USER_EMAIL_HEADER, "").strip()

        user_id = None
        if raw_user_id:
            try:
                user_id = int(raw_user_id)
            except (TypeError, ValueError) as exc:
                raise AuthenticationFailed(
                    f"{USER_ID_HEADER} must be a positive integer."
                ) from exc

            if user_id <= 0:
                raise AuthenticationFailed(
                    f"{USER_ID_HEADER} must be a positive integer."
                )

        if raw_email:
            try:
                validate_email(raw_email)
            except ValidationError as exc:
                raise AuthenticationFailed(
                    f"{USER_EMAIL_HEADER} must be a valid email address."
                ) from exc

        return SRSServiceCaller(user_id=user_id, email=raw_email)


def caller_audit_fields(request, id_field, email_field):
    """Build model values from the authenticated caller's audit headers."""

    caller = request.user
    return {
        id_field: getattr(caller, "user_id", None),
        email_field: getattr(caller, "email", ""),
    }


class SRSSharedKeyAuthenticationScheme(OpenApiAuthenticationExtension):
    """Show the shared-key header in Swagger's Authorize dialog."""

    target_class = "srs.authentication.SRSSharedKeyAuthentication"
    name = "SRSApiKey"

    def get_security_definition(self, auto_schema):
        return {
            "type": "apiKey",
            "in": "header",
            "name": API_KEY_HEADER,
            "description": "Server-to-server key shared with the SRS website.",
        }
