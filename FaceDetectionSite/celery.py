from __future__ import absolute_import, unicode_literals

import os

from celery import Celery

# set the default Django settings module for the 'celery' program.
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'FaceDetectionSite.settings')

clry_app = Celery('detect_tasks')

# Using a string here means the worker doesn't have to serialize
# the configuration object to child processes.
# - namespace='CELERY' means all celery-related configuration keys
#   should have a `CELERY_` prefix.
clry_app.config_from_object('django.conf:settings', namespace='CELERY')

clry_app.conf.update(CELERY_ACCEPT_CONTENT = ['pickle', 'json'])
clry_app.conf.update(CELERY_TASK_SERIALIZER = ['pickle'])
clry_app.conf.update(CELERY_RESULT_SERIALIZER = ['pickle'])

# Load task modules from all registered Django app configs.
clry_app.autodiscover_tasks()