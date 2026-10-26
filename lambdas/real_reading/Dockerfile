# Real reading Lambda (contract 2.0). Build context is assembled by assemble.sh.
FROM public.ecr.aws/lambda/python:3.12
RUN dnf install -y mesa-libGL dejavu-sans-fonts && dnf clean all
COPY requirements.txt ${LAMBDA_TASK_ROOT}/
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt
COPY app/ ${LAMBDA_TASK_ROOT}/
CMD ["lambda_v1.handler"]
