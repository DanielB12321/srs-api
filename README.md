# SRS API

Django REST API for the Signature Reference System. It stores uploaded datasets,
the OSNACA reference library and ranked similarity analyses.

## Local setup

Python 3.11 is the supported runtime.

```powershell
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env_defaults .env
python manage.py migrate
python manage.py runserver
```

Set these values in `.env` before starting the API:

```env
SECRET_KEY=replace-with-a-local-django-key
SRS_API_SHARED_KEY=replace-with-the-key-used-by-SRS_QUT
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
CORS_ALLOWED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000
```

SQLite is used locally when `DATABASE_URL` is not set. Deployed environments
should provide a PostgreSQL `DATABASE_URL`, `DEBUG=False` and their public host
name in `ALLOWED_HOSTS`.

## API authentication and documentation

Functional endpoints require the shared key in this header:

```text
X-SRS-API-Key: your-shared-key
```

SRS_QUT adds the header to its server-side requests. Any client that has the key
can call the API, so the key must not be placed in browser JavaScript or committed
to Git. Optional `X-SRS-User-ID` and `X-SRS-User-Email` headers are stored for
auditing only; records remain shared between SRS_QUT users.

The public documentation pages are:

- Swagger UI: `http://127.0.0.1:8000/api/docs/`
- OpenAPI schema: `http://127.0.0.1:8000/api/schema/`

In Swagger, select **Authorize** and enter the raw shared key for `SRSApiKey`.

## Dataset CSV format

Dataset uploads currently accept CSV files. A sample identifier is required;
latitude and longitude are optional. Measurement headings use
`Element_unit`, for example:

```csv
sample_id,latitude,longitude,Cu_ppm,Au_ppm
S001,-27.47,153.03,120.5,0.42
```

Blank and non-numeric measurements are stored as missing values. `NaN` and
infinite values are rejected. Coordinates, when supplied, must use normal
latitude and longitude ranges.

## Reference-library commands

Import a new OSNACA workbook pair:

```powershell
python manage.py import_osnaca --data "OSNACA-Data-1.xlsx" --metadata "OSNACA-Metadata-1.xlsx" --source-name "OSNACA v1"
```

Re-run a stored import or seed lookup tables separately:

```powershell
python manage.py import_osnaca --rerun 7
python manage.py seed_reference_lookups --metadata "OSNACA-Metadata-1.xlsx"
```

Other offline tools are available through `benchmark_algorithms`,
`build_projection` and `train_ml_ensemble`. Run a command with `--help` for its
arguments.

## Checks and tests

```powershell
python manage.py check
python manage.py test
```

The Azure workflow installs dependencies and runs both commands before it
deploys the `main` branch.

## Azure deployment

Configure `SECRET_KEY`, `SRS_API_SHARED_KEY`, `DATABASE_URL`, `ALLOWED_HOSTS` and
`CORS_ALLOWED_ORIGINS` in Azure App Settings. Use the same shared API key in the
SRS_QUT deployment, then restart both applications.

Run `python manage.py migrate` against the deployment database whenever new
migrations are released. Uploaded workbooks also need persistent media storage
if imports must be rerun later.

Reference imports and full analyses currently run in the web process. Allow
running jobs to finish before restarting or redeploying the API.
