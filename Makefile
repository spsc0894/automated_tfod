setup:
	mkdir Tensorflow
	cd Tensorflow && git clone https://github.com/tensorflow/models.git
	git clone https://github.com/cocodataset/cocoapi.git
	docker build -t activeeon/tensorboard ./tb/

run_tfod:
	xhost +
	docker-compose up -d
stop_tfod:
	docker-compose down
tensorboard:
	docker run -v automated_tfod_shared:/logs activeeon/tensorboard:latest
