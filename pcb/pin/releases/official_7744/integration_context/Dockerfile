FROM eda-pin-l46-t:submission-20261015
COPY payload/ /submission/
RUN --mount=type=bind,source=wheels,target=/integration_wheels,readonly PYTHONDONTWRITEBYTECODE=1 PIP_DISABLE_PIP_VERSION_CHECK=1 python3 -m pip install --no-index --find-links=/integration_wheels --no-deps --no-build-isolation --no-cache-dir --no-compile --target=/submission/.runtime/alternative_frontend rapidocr==3.9.2 omegaconf==2.3.0 colorlog==6.10.1 antlr4-python3-runtime==4.9.3 && python3 -B /submission/integration_runtime/finalize_manifest.py && chmod +x /submission/run.sh
ENTRYPOINT ["/submission/run.sh"]
