FROM public.ecr.aws/lambda/python:3.11
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY lambda_handler.py ${LAMBDA_TASK_ROOT}/
CMD ["lambda_handler.lambda_handler"]
