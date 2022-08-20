import tensorflow as tf
import tensorflow_data_validation as tfdv

from tfx import v1 as tfx

from tfx.types import standard_artifacts

from tfx.orchestration.experimental.interactive.interactive_context import InteractiveContext
from google.protobuf.json_format import MessageToDict
from tensorflow_metadata.proto.v0 import schema_pb2

import os
import pprint
pp = pprint.PrettyPrinter()


# ### Define paths
# 
# For familiarity, you will again be using the [Census Income dataset](https://archive.ics.uci.edu/ml/datasets/Adult) from the previous weeks' ungraded labs. You will use the same paths to your raw data and pipeline files as shown below.

# In[2]:


# location of the pipeline metadata store
_pipeline_root = './pipeline/'

# directory of the raw data files
_data_root = './data/census_data'

# path to the raw training data
_data_filepath = os.path.join(_data_root, 'adult.data')


# ## Data Pipeline
# 
# Each TFX component you use accepts and generates artifacts which are instances of the different artifact types TFX has configured in the metadata store. The properties of these instances are shown neatly in a table in the outputs of `context.run()`. TFX does all of these for you so you only need to inspect the output of each component to know which property of the artifact you can pass on to the next component (e.g. the `outputs['examples']` of `ExampleGen` can be passed to `StatisticsGen`).
# 
# Since you've already used this dataset before, we will just quickly go over `ExampleGen`, `StatisticsGen`, and `SchemaGen`. The new concepts will be discussed after the said components.

# ### Create the Interactive Context

# In[3]:


# Initialize the InteractiveContext.
# If you leave `_pipeline_root` blank, then the db will be created in a temporary directory.
context = InteractiveContext(pipeline_root=_pipeline_root)


# ### ExampleGen

# In[4]:


# Instantiate ExampleGen with the input CSV dataset
example_gen = tfx.components.CsvExampleGen(input_base=_data_root)

# Execute the component
context.run(example_gen)


# ### StatisticsGen

# In[5]:


# Instantiate StatisticsGen with the ExampleGen ingested dataset
statistics_gen = tfx.components.StatisticsGen(
    examples=example_gen.outputs['examples'])

# Execute the component
context.run(statistics_gen)


# ### SchemaGen

# In[6]:


# Instantiate SchemaGen with the StatisticsGen ingested dataset
schema_gen = tfx.components.SchemaGen(
    statistics=statistics_gen.outputs['statistics'],
    )

# Run the component
context.run(schema_gen)


# In[7]:


# Visualize the schema
context.show(schema_gen.outputs['schema'])


# ### Curating the Schema

# Now that you have the inferred schema, you can proceed to revising it to be more robust. For instance, you can restrict the age as you did in Week 1. First, you have to load the `Schema` protocol buffer from the metadata store. You can do this by getting the schema uri from the output of `SchemaGen` then use TFDV's `load_schema_text()` method.

# In[8]:


# Get the schema uri
schema_uri = schema_gen.outputs['schema']._artifacts[0].uri

# Get the schema pbtxt file from the SchemaGen output
schema = tfdv.load_schema_text(os.path.join(schema_uri, 'schema.pbtxt'))


# With that, you can now make changes to the schema as before. For the purpose of this exercise, you will only modify the age domain but feel free to add more if you want.

# In[9]:


# Restrict the range of the `age` feature
tfdv.set_domain(schema, 'age', schema_pb2.IntDomain(name='age', min=17, max=90))

# Display the modified schema. Notice the `Domain` column of `age`.
tfdv.display_schema(schema)


# ### Schema Environments
# 
# By default, your schema will watch for all the features declared above including the label. However, when the model is served for inference, it will get datasets that will not have the label because that is the feature that the model will be trying to predict. You need to configure the pipeline to not raise an alarm when this kind of dataset is received.
# 
# You can do that with [schema environments](https://www.tensorflow.org/tfx/tutorials/data_validation/tfdv_basic#schema_environments). First, you will need to declare training and serving environments, then configure the serving schema to not watch for the presence of labels. See how it is implemented below.

# In[10]:


# Create schema environments for training and serving
schema.default_environment.append('TRAINING')
schema.default_environment.append('SERVING')

# Omit label from the serving environment
tfdv.get_feature(schema, 'label').not_in_environment.append('SERVING')


# You can now freeze the curated schema and save to a local directory.

# In[11]:


# Declare the path to the updated schema directory
_updated_schema_dir = f'{_pipeline_root}/updated_schema'

# Declare the path to the schema file
schema_file = os.path.join(_updated_schema_dir, 'schema.pbtxt')

# Save the curated schema to the said file
tfdv.write_schema_text(schema, schema_file)


# ### ImportSchemaGen
# 
# Now that the schema has been saved, you need to create an artifact in the metadata store that will point to it. TFX provides the [ImporterSchemaGen](https://www.tensorflow.org/tfx/api_docs/python/tfx/v1/components/ImportSchemaGen) component used to import a curated schema to ML Metadata. You simply need to specify the URL of the revised schema file.

# In[12]:


# Use ImportSchemaGen to put the curated schema to ML Metadata
user_schema_importer = tfx.components.ImportSchemaGen(schema_file=schema_file)

# Run the component
context.run(user_schema_importer, enable_cache=False)


# If you pass in the component output to `context.show()`, then you should see the schema.

# In[13]:


# See the result
context.show(user_schema_importer.outputs['schema'])


# ### ExampleValidator
# 
# You can then use this new artifact as input to the other components of the pipeline. See how it is used as the `schema` argument in `ExampleValidator` below.

# In[14]:


# Instantiate ExampleValidator with the StatisticsGen and SchemaGen ingested data
example_validator = tfx.components.ExampleValidator(
    statistics=statistics_gen.outputs['statistics'],
    schema=user_schema_importer.outputs['schema'])

# Run the component.
context.run(example_validator)


# In[15]:


# Visualize the results
context.show(example_validator.outputs['anomalies'])


# ### Practice with ML Metadata
# 
# At this point, you should now take some time exploring the contents of the metadata store saved by your component runs. This will let you practice tracking artifacts and how they are related to each other. This involves looking at artifacts, executions, and events. This skill will let you recover related artifacts even without seeing the code of the training run. All you need is access to the metadata store. 
# 
# See how the input artifact IDs to an instance of `ExampleAnomalies` are tracked in the following cells. If you have this notebook, then you will already know that it uses the output of StatisticsGen for this run and also the curated schema you imported. However, if you already have hundreds of training runs and parameter iterations, you may find it hard to track which is which. That's where the metadata store will be useful. Since it records information about a specific pipeline run, you will be able to track the inputs and outputs of a particular artifact.
# 
# You will start by setting the connection config to the metadata store. 

# In[16]:


# Import mlmd and utilities
import ml_metadata as mlmd
from ml_metadata.proto import metadata_store_pb2

# Get the connection config to connect to the context's metadata store
connection_config = context.metadata_connection_config

# Instantiate a MetadataStore instance with the connection config
store = mlmd.MetadataStore(connection_config)


# Next, let's see what artifact types are available in the metadata store.

# In[17]:


# Get artifact types
artifact_types = store.get_artifact_types()

# Print the results
[artifact_type.name for artifact_type in artifact_types]


# If you get the artifacts of type `Schema`, you will see that there are two entries. One is the inferred and the other is the one you imported. At the end of this exercise, you can verify that the curated schema is the one used for the `ExampleValidator` run we will be investigating.

# In[18]:


# Get artifact types
schema_list = store.get_artifacts_by_type('Schema')

[(f'schema uri: {schema.uri}', f'schema id:{schema.id}') for schema in schema_list]


# Let's get the first instance of `ExampleAnomalies` to get the output of `ExampleValidator`.

# In[19]:


# Get 1st instance of ExampleAnomalies
example_anomalies = store.get_artifacts_by_type('ExampleAnomalies')[0]

# Print the artifact id
print(f'Artifact id: {example_anomalies.id}')


# You will use the artifact ID to get events related to it. Let's just get the first instance.

# In[20]:


# Get first event related to the ID
anomalies_id_event = store.get_events_by_artifact_ids([example_anomalies.id])[0]

# Print results
print(anomalies_id_event)


# As expected, the event type will be an `OUTPUT` because this is the output of the `ExampleValidator` component. Since we want to get the inputs, we can track it through the execution id.

# In[21]:


# Get execution ID
anomalies_execution_id = anomalies_id_event.execution_id

# Get events by the execution ID
events_execution = store.get_events_by_execution_ids([anomalies_execution_id])

# Print results
print(events_execution)


# We see the artifacts which are marked as `INPUT` above representing the statistics and schema inputs. We can extract their IDs programmatically like this. You will see that you will get the artifact ID of the curated schema you printed out earlier.

# In[22]:


# Filter INPUT type events
inputs_to_exval = [event.artifact_id for event in events_execution 
                       if event.type == metadata_store_pb2.Event.INPUT]

# Print results
print(inputs_to_exval)


# **Congratulations!** You have now completed this notebook on iterative schemas and saw how it can be used in a TFX pipeline. You were also able to track an artifact's lineage by looking at the artifacts, events, and executions in the metadata store. These will come in handy in this week's assignment!
