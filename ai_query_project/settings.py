from pathlib import Path
import os 
from pathlib import Path
import os
from pymongo import MongoClient
from dotenv import load_dotenv
import json
from os import path
import json
import pickle
from pymongo import MongoClient
import pandas as pd
import joblib
# Load environment variables from .env file
load_dotenv()

# MongoDB connection using environment variables
mongo_uri = "mongodb://3402f86a8f1d7349340f6e2b155c193f90ef8d09a8287e960ee7dc46152bc23f:e4073cb35bd6a9f2219050739d4b2e3831e3e8a535533d8e557ee939399469fc@13.203.49.68:27720/?authMechanism=DEFAULT&authSource=admin"
client = MongoClient(mongo_uri)
db = client[os.getenv('MONGO_DB')]
COLLECTION = db[os.getenv('MONGO_COLLECTION')]
# with open('knn_model_tfidf_only_ind.pkl', 'rb') as f:
#     KNN_MODEL = pickle.load(f)
# with open('uuid_tfidf_only_ind.pkl', 'rb') as f:
#     UUID_LIST = pickle.load(f)
#LOADED_TFIDF = joblib.load("tfidf_vectorizer_cleaned_ind2.pkl") # prev tfidf_vectorizer_cleaned.pkl 



BASE_DIR = Path(__file__).resolve().parent.parent

# Load the FAISS index and company DataFrame
# FAISS_INDEX_PATH = os.path.join(BASE_DIR, ".", "data", "faiss_index_live.bin")
#PICKLE_CHUNKS_DIR = os.path.join(BASE_DIR, ".","data")



# Build paths inside the project like this: BASE_DIR / 'subdir'.


# Quick-start development settings - unsuitable for production
# See https://docs.djangoproject.com/en/5.1/howto/deployment/checklist/

# SECURITY WARNING: keep the secret key used in production secret!
SECRET_KEY = 'django-insecure-9*+jquj*kgrv!yj&r^%ok(ws93l(ioem5kon0l6=@7cr$3%lyo'

# SECURITY WARNING: don't run with debug turned on in production!
DEBUG = False

ALLOWED_HOSTS = ["18.60.242.203", "localhost", "127.0.0.1", "0.0.0.0", "18.61.173.37",'98.130.54.55']
CORS_ALLOW_ALL_ORIGINS = True

CORS_ALLOWED_ORIGINS = [
    "http://localhost:5003",  # If your frontend runs on localhost:3000
]
# Application definition

INSTALLED_APPS = [
    'django.contrib.admin',
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'corsheaders'
]

MIDDLEWARE = [
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',
    'corsheaders.middleware.CorsMiddleware',
    'searchapi.middleware.APIRequestMiddleware',
    
]

ROOT_URLCONF = 'ai_query_project.urls'

TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]

WSGI_APPLICATION = 'ai_query_project.wsgi.application'


# Database
# https://docs.djangoproject.com/en/5.1/ref/settings/#databases

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


# Password validation
# https://docs.djangoproject.com/en/5.1/ref/settings/#auth-password-validators

AUTH_PASSWORD_VALIDATORS = [
    {
        'NAME': 'django.contrib.auth.password_validation.UserAttributeSimilarityValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.MinimumLengthValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.CommonPasswordValidator',
    },
    {
        'NAME': 'django.contrib.auth.password_validation.NumericPasswordValidator',
    },
]


# Internationalization
# https://docs.djangoproject.com/en/5.1/topics/i18n/

LANGUAGE_CODE = 'en-us'

TIME_ZONE = 'UTC'

USE_I18N = True

USE_TZ = True


# Static files (CSS, JavaScript, Images)
# https://docs.djangoproject.com/en/5.1/howto/static-files/

STATIC_URL = 'static/'

# Default primary key field type
# https://docs.djangoproject.com/en/5.1/ref/settings/#default-auto-field

DEFAULT_AUTO_FIELD = 'django.db.models.BigAutoField'
