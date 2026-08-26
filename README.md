# srs-api

Backend for the Signature Reference System. It handles dataset uploads,
reference data and similarity analyses.

## API authentication

Functional API endpoints accept calls only from the SRS_QUT web server. Set
`SRS_API_SHARED_KEY` in this application's `.env` or Azure App Settings, and
set the same value in SRS_QUT. Generate a suitable value with:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

The web server sends the key in `X-SRS-API-Key`. Requests fail with `401` when
the key is absent, wrong, or not configured on the API. Never put this key in
browser JavaScript.

`X-SRS-User-ID` and `X-SRS-User-Email` are optional audit headers. When present,
they are saved on new dataset, reference-import and full-analysis records. They
do not control access: SRS data remains shared by all authenticated SRS_QUT
users, including records created before these headers were introduced.

The schema and Swagger page remain public at `/api/schema/` and `/api/docs/`.
Use Swagger's **Authorize** button to provide the shared key before trying a
functional endpoint.

See `.env_defaults` for the local environment variable names.
