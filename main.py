import boto3
s3 = boto3.client('s3')
s3.upload_file(file_name, bucket, object_name=None)

fraud_df = pd.read_csv(<your S3 file location>)

X = fraud_df[set(fraud_df.columns) - set(['Class'])]
y = fraud_df['Class']
print (y.value_counts())

auto_ml_job_name = 'automl-creditcard-fraud'
import boto3
sm = boto3.client('sagemaker')
import sagemaker  
session = sagemaker.Session()

prefix = 'sagemaker/' + auto_ml_job_name
bucket = session.default_bucket()
training_data = pd.DataFrame(X_train)
training_data['Class'] = list(y_train)
test_data = pd.DataFrame(X_test)

train_file = 'train_data.csv';
training_data.to_csv(train_file, index=False, header=True)
train_data_s3_path = session.upload_data(path=train_file, key_prefix=prefix + "/train")
print('Train data uploaded to: ' + train_data_s3_path)

test_file = 'test_data.csv';
test_data.to_csv(test_file, index=False, header=False)
test_data_s3_path = session.upload_data(path=test_file, key_prefix=prefix + "/test")
print('Test data uploaded to: ' + test_data_s3_path)
input_data_config = [{
      'DataSource': {
        'S3DataSource': {
          'S3DataType': 'S3Prefix',
          'S3Uri': 's3://{}/{}/train'.format(bucket,prefix)
        }
      },
      'TargetAttributeName': 'Class'
    }
  ]
  
from sagemaker.automl.automl import AutoML
from time import gmtime, strftime, sleep
from sagemaker import get_execution_role

timestamp_suffix = strftime('%d-%H-%M-%S', gmtime())
base_job_name = 'automl-card-fraud' 

target_attribute_name = 'Class'
role = get_execution_role()
automl = AutoML(role=role,
                target_attribute_name=target_attribute_name,
                base_job_name=base_job_name,
                sagemaker_session=session,
                problem_type='BinaryClassification',
                job_objective={'MetricName': 'AUC'},
                max_candidates=100)                
                
automl.fit(train_file, job_name=base_job_name, wait=False, logs=False)
describe_response = automl.describe_auto_ml_job()
print (describe_response)
job_run_status = describe_response['AutoMLJobStatus']
    
while job_run_status not in ('Failed', 'Completed', 'Stopped'):
    describe_response = automl.describe_auto_ml_job()
    job_run_status = describe_response['AutoMLJobStatus']
    print (job_run_status)
    sleep(30)
print ('completed')

best_candidate = automl.describe_auto_ml_job()['BestCandidate']
best_candidate_name = best_candidate['CandidateName']
print("CandidateName: " + best_candidate_name)
print("FinalAutoMLJobObjectiveMetricName: " + best_candidate['FinalAutoMLJobObjectiveMetric']['MetricName'])
print("FinalAutoMLJobObjectiveMetricValue: " + str(best_candidate['FinalAutoMLJobObjectiveMetric']['Value']))
