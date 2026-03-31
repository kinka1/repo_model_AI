-- Database: TA KITA

-- DROP DATABASE IF EXISTS "TA KITA";

CREATE DATABASE "TA KITA"
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'Indonesian_Indonesia.1252'
    LC_CTYPE = 'Indonesian_Indonesia.1252'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

-- Table: public.ai_models

-- DROP TABLE IF EXISTS public.ai_models;

CREATE TABLE IF NOT EXISTS public.ai_models
(
    id integer NOT NULL DEFAULT nextval('ai_models_id_seq'::regclass),
    model_name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    model_type character varying(50) COLLATE pg_catalog."default" NOT NULL,
    version character varying(20) COLLATE pg_catalog."default" NOT NULL,
    model_file_path character varying(255) COLLATE pg_catalog."default",
    accuracy numeric(5,4),
    precision_score numeric(5,4),
    recall_score numeric(5,4),
    f1_score numeric(5,4),
    is_active boolean DEFAULT false,
    created_by_user_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT ai_models_pkey PRIMARY KEY (id),
    CONSTRAINT ai_models_model_name_version_key UNIQUE (model_name, version),
    CONSTRAINT ai_models_created_by_user_id_fkey FOREIGN KEY (created_by_user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.ai_models
    OWNER to postgres;
-- Index: idx_ai_models_active

-- DROP INDEX IF EXISTS public.idx_ai_models_active;

CREATE INDEX IF NOT EXISTS idx_ai_models_active
    ON public.ai_models USING btree
    (is_active ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_ai_models_name_version

-- DROP INDEX IF EXISTS public.idx_ai_models_name_version;

CREATE INDEX IF NOT EXISTS idx_ai_models_name_version
    ON public.ai_models USING btree
    (model_name COLLATE pg_catalog."default" ASC NULLS LAST, version COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_one_active_model

-- DROP INDEX IF EXISTS public.idx_one_active_model;

CREATE UNIQUE INDEX IF NOT EXISTS idx_one_active_model
    ON public.ai_models USING btree
    (is_active ASC NULLS LAST)
    TABLESPACE pg_default
    WHERE is_active = true;

-- Trigger: update_ai_models_updated_at

-- DROP TRIGGER IF EXISTS update_ai_models_updated_at ON public.ai_models;

CREATE OR REPLACE TRIGGER update_ai_models_updated_at
    BEFORE UPDATE 
    ON public.ai_models
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Table: public.audit_logs

-- DROP TABLE IF EXISTS public.audit_logs;

CREATE TABLE IF NOT EXISTS public.audit_logs
(
    id integer NOT NULL DEFAULT nextval('audit_logs_id_seq'::regclass),
    user_id integer,
    action character varying(100) COLLATE pg_catalog."default" NOT NULL,
    table_name character varying(50) COLLATE pg_catalog."default",
    record_id integer,
    old_value jsonb,
    new_value jsonb,
    ip_address character varying(45) COLLATE pg_catalog."default",
    user_agent text COLLATE pg_catalog."default",
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT audit_logs_pkey PRIMARY KEY (id),
    CONSTRAINT audit_logs_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.audit_logs
    OWNER to postgres;
-- Index: idx_audit_action

-- DROP INDEX IF EXISTS public.idx_audit_action;

CREATE INDEX IF NOT EXISTS idx_audit_action
    ON public.audit_logs USING btree
    (action COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_audit_date

-- DROP INDEX IF EXISTS public.idx_audit_date;

CREATE INDEX IF NOT EXISTS idx_audit_date
    ON public.audit_logs USING btree
    (created_at ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_audit_user

-- DROP INDEX IF EXISTS public.idx_audit_user;

CREATE INDEX IF NOT EXISTS idx_audit_user
    ON public.audit_logs USING btree
    (user_id ASC NULLS LAST)
    TABLESPACE pg_default;

-- Table: public.classifications

-- DROP TABLE IF EXISTS public.classifications;

CREATE TABLE IF NOT EXISTS public.classifications
(
    id integer NOT NULL DEFAULT nextval('classifications_id_seq'::regclass),
    patient_id integer NOT NULL,
    dataset_id integer,
    image_file_name character varying(255) COLLATE pg_catalog."default" NOT NULL,
    image_path character varying(500) COLLATE pg_catalog."default" NOT NULL,
    classified_by_model_id integer,
    classification_gram character varying(20) COLLATE pg_catalog."default",
    classification_bentuk character varying(20) COLLATE pg_catalog."default",
    confidence_score numeric(5,4),
    validation_gram character varying(20) COLLATE pg_catalog."default",
    validation_bentuk character varying(20) COLLATE pg_catalog."default",
    catatan_dokter text COLLATE pg_catalog."default",
    reannotated_by_user_id integer,
    reannotated_at timestamp without time zone,
    classified_by_user_id integer,
    classified_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT classifications_pkey PRIMARY KEY (id),
    CONSTRAINT classifications_classified_by_model_id_fkey FOREIGN KEY (classified_by_model_id)
        REFERENCES public.ai_models (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT classifications_classified_by_user_id_fkey FOREIGN KEY (classified_by_user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT classifications_dataset_id_fkey FOREIGN KEY (dataset_id)
        REFERENCES public.datasets (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT classifications_patient_id_fkey FOREIGN KEY (patient_id)
        REFERENCES public.patients (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE,
    CONSTRAINT classifications_reannotated_by_user_id_fkey FOREIGN KEY (reannotated_by_user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT classifications_classification_gram_check CHECK (classification_gram::text = ANY (ARRAY['Positif'::character varying, 'Negatif'::character varying]::text[])),
    CONSTRAINT classifications_classification_bentuk_check CHECK (classification_bentuk::text = ANY (ARRAY['Batang'::character varying, 'Kokus'::character varying]::text[])),
    CONSTRAINT classifications_validation_gram_check CHECK (validation_gram::text = ANY (ARRAY['Positif'::character varying, 'Negatif'::character varying]::text[])),
    CONSTRAINT classifications_validation_bentuk_check CHECK (validation_bentuk::text = ANY (ARRAY['Batang'::character varying, 'Kokus'::character varying]::text[]))
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.classifications
    OWNER to postgres;
-- Index: idx_classifications_date

-- DROP INDEX IF EXISTS public.idx_classifications_date;

CREATE INDEX IF NOT EXISTS idx_classifications_date
    ON public.classifications USING btree
    (classified_at ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_classifications_model

-- DROP INDEX IF EXISTS public.idx_classifications_model;

CREATE INDEX IF NOT EXISTS idx_classifications_model
    ON public.classifications USING btree
    (classified_by_model_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_classifications_patient

-- DROP INDEX IF EXISTS public.idx_classifications_patient;

CREATE INDEX IF NOT EXISTS idx_classifications_patient
    ON public.classifications USING btree
    (patient_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_classifications_pending

-- DROP INDEX IF EXISTS public.idx_classifications_pending;

CREATE INDEX IF NOT EXISTS idx_classifications_pending
    ON public.classifications USING btree
    (patient_id ASC NULLS LAST, validation_gram COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default
    WHERE validation_gram IS NULL;
-- Index: idx_classifications_validated

-- DROP INDEX IF EXISTS public.idx_classifications_validated;

CREATE INDEX IF NOT EXISTS idx_classifications_validated
    ON public.classifications USING btree
    (patient_id ASC NULLS LAST, validation_gram COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default
    WHERE validation_gram IS NOT NULL;

-- Trigger: update_classifications_updated_at

-- DROP TRIGGER IF EXISTS update_classifications_updated_at ON public.classifications;

CREATE OR REPLACE TRIGGER update_classifications_updated_at
    BEFORE UPDATE 
    ON public.classifications
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Table: public.datasets

-- DROP TABLE IF EXISTS public.datasets;

CREATE TABLE IF NOT EXISTS public.datasets
(
    id integer NOT NULL DEFAULT nextval('datasets_id_seq'::regclass),
    dataset_name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    dataset_file_path character varying(255) COLLATE pg_catalog."default",
    total_images integer DEFAULT 0,
    gram_positive_count integer DEFAULT 0,
    gram_negative_count integer DEFAULT 0,
    uploaded_by_user_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT datasets_pkey PRIMARY KEY (id),
    CONSTRAINT datasets_uploaded_by_user_id_fkey FOREIGN KEY (uploaded_by_user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.datasets
    OWNER to postgres;

-- Table: public.model_training_status

-- DROP TABLE IF EXISTS public.model_training_status;

CREATE TABLE IF NOT EXISTS public.model_training_status
(
    id integer NOT NULL DEFAULT nextval('model_training_status_id_seq'::regclass),
    model_id integer,
    dataset_id integer,
    status character varying(20) COLLATE pg_catalog."default" NOT NULL,
    progress numeric(3,2) DEFAULT 0.0,
    start_time timestamp without time zone,
    end_time timestamp without time zone,
    error_message text COLLATE pg_catalog."default",
    current_epoch integer,
    total_epochs integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT model_training_status_pkey PRIMARY KEY (id),
    CONSTRAINT model_training_status_dataset_id_fkey FOREIGN KEY (dataset_id)
        REFERENCES public.datasets (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT model_training_status_model_id_fkey FOREIGN KEY (model_id)
        REFERENCES public.ai_models (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE,
    CONSTRAINT model_training_status_status_check CHECK (status::text = ANY (ARRAY['TRAINING'::character varying, 'COMPLETED'::character varying, 'FAILED'::character varying, 'IDLE'::character varying]::text[]))
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.model_training_status
    OWNER to postgres;
-- Index: idx_training_model_id

-- DROP INDEX IF EXISTS public.idx_training_model_id;

CREATE INDEX IF NOT EXISTS idx_training_model_id
    ON public.model_training_status USING btree
    (model_id ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_training_status

-- DROP INDEX IF EXISTS public.idx_training_status;

CREATE INDEX IF NOT EXISTS idx_training_status
    ON public.model_training_status USING btree
    (status COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

-- Table: public.patients

-- DROP TABLE IF EXISTS public.patients;

CREATE TABLE IF NOT EXISTS public.patients
(
    id integer NOT NULL DEFAULT nextval('patients_id_seq'::regclass),
    id_pasien character varying(20) COLLATE pg_catalog."default" NOT NULL,
    nama_lengkap character varying(100) COLLATE pg_catalog."default" NOT NULL,
    jenis_kelamin character varying(20) COLLATE pg_catalog."default" NOT NULL,
    tanggal_lahir date NOT NULL,
    alamat text COLLATE pg_catalog."default",
    no_telepon character varying(20) COLLATE pg_catalog."default",
    created_by_user_id integer,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT patients_pkey PRIMARY KEY (id),
    CONSTRAINT patients_id_pasien_key UNIQUE (id_pasien),
    CONSTRAINT patients_created_by_user_id_fkey FOREIGN KEY (created_by_user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE NO ACTION,
    CONSTRAINT patients_jenis_kelamin_check CHECK (jenis_kelamin::text = ANY (ARRAY['Laki-Laki'::character varying, 'Perempuan'::character varying]::text[]))
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.patients
    OWNER to postgres;
-- Index: idx_patients_created_at

-- DROP INDEX IF EXISTS public.idx_patients_created_at;

CREATE INDEX IF NOT EXISTS idx_patients_created_at
    ON public.patients USING btree
    (created_at ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_patients_id_pasien

-- DROP INDEX IF EXISTS public.idx_patients_id_pasien;

CREATE INDEX IF NOT EXISTS idx_patients_id_pasien
    ON public.patients USING btree
    (id_pasien COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_patients_nama

-- DROP INDEX IF EXISTS public.idx_patients_nama;

CREATE INDEX IF NOT EXISTS idx_patients_nama
    ON public.patients USING btree
    (nama_lengkap COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

-- Trigger: update_patients_updated_at

-- DROP TRIGGER IF EXISTS update_patients_updated_at ON public.patients;

CREATE OR REPLACE TRIGGER update_patients_updated_at
    BEFORE UPDATE 
    ON public.patients
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();

-- Table: public.sessions

-- DROP TABLE IF EXISTS public.sessions;

CREATE TABLE IF NOT EXISTS public.sessions
(
    id integer NOT NULL DEFAULT nextval('sessions_id_seq'::regclass),
    user_id integer NOT NULL,
    token_hash character varying(255) COLLATE pg_catalog."default" NOT NULL,
    expires_at timestamp without time zone NOT NULL,
    is_revoked boolean DEFAULT false,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT sessions_pkey PRIMARY KEY (id),
    CONSTRAINT sessions_token_hash_key UNIQUE (token_hash),
    CONSTRAINT sessions_user_id_fkey FOREIGN KEY (user_id)
        REFERENCES public.users (id) MATCH SIMPLE
        ON UPDATE NO ACTION
        ON DELETE CASCADE
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.sessions
    OWNER to postgres;
-- Index: idx_sessions_expires

-- DROP INDEX IF EXISTS public.idx_sessions_expires;

CREATE INDEX IF NOT EXISTS idx_sessions_expires
    ON public.sessions USING btree
    (expires_at ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_sessions_token

-- DROP INDEX IF EXISTS public.idx_sessions_token;

CREATE INDEX IF NOT EXISTS idx_sessions_token
    ON public.sessions USING btree
    (token_hash COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_sessions_user

-- DROP INDEX IF EXISTS public.idx_sessions_user;

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON public.sessions USING btree
    (user_id ASC NULLS LAST)
    TABLESPACE pg_default;

-- Table: public.users

-- DROP TABLE IF EXISTS public.users;

CREATE TABLE IF NOT EXISTS public.users
(
    id integer NOT NULL DEFAULT nextval('users_id_seq'::regclass),
    username character varying(50) COLLATE pg_catalog."default" NOT NULL,
    email character varying(100) COLLATE pg_catalog."default" NOT NULL,
    hashed_password character varying(255) COLLATE pg_catalog."default" NOT NULL,
    full_name character varying(100) COLLATE pg_catalog."default" NOT NULL,
    role character varying(20) COLLATE pg_catalog."default" NOT NULL,
    is_active boolean DEFAULT true,
    created_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    updated_at timestamp without time zone DEFAULT CURRENT_TIMESTAMP,
    last_login timestamp without time zone,
    CONSTRAINT users_pkey PRIMARY KEY (id),
    CONSTRAINT users_email_key UNIQUE (email),
    CONSTRAINT users_username_key UNIQUE (username),
    CONSTRAINT users_role_check CHECK (role::text = ANY (ARRAY['Admin'::character varying, 'Analis'::character varying, 'Dokter'::character varying]::text[]))
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.users
    OWNER to postgres;
-- Index: idx_users_email

-- DROP INDEX IF EXISTS public.idx_users_email;

CREATE INDEX IF NOT EXISTS idx_users_email
    ON public.users USING btree
    (email COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_users_role

-- DROP INDEX IF EXISTS public.idx_users_role;

CREATE INDEX IF NOT EXISTS idx_users_role
    ON public.users USING btree
    (role COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;
-- Index: idx_users_username

-- DROP INDEX IF EXISTS public.idx_users_username;

CREATE INDEX IF NOT EXISTS idx_users_username
    ON public.users USING btree
    (username COLLATE pg_catalog."default" ASC NULLS LAST)
    TABLESPACE pg_default;

-- Trigger: update_users_updated_at

-- DROP TRIGGER IF EXISTS update_users_updated_at ON public.users;

CREATE OR REPLACE TRIGGER update_users_updated_at
    BEFORE UPDATE 
    ON public.users
    FOR EACH ROW
    EXECUTE FUNCTION public.update_updated_at_column();