## Kafka configs
.\config\server.properties
process.roles=broker,controller
node.id=1
controller.quorum.voters=1@localhost:9093
listeners=PLAINTEXT://localhost:9092,CONTROLLER://localhost:9093
log.dirs=logs

## Generation kafaka cluster
.\bin\windows\kafka-storage.bat format -t %RANDOM% -c .\config\kraft\server.properties

## Run Kafka broker
.\bin\windows\kafka-server-start.bat .\config\server.properties

## List of topics
bin\windows\kafka-topics.bat --bootstrap-server localhost:9092 --list

## Create topic
.\bin\windows\kafka-topics.bat --create --topic tweets --bootstrap-server localhost:9092 --partitions 1 --replication-factor 1

## Install and lunch project
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python run_producer.py
python run_processor.py
python run_analyzer.py
python run_dashboard.py