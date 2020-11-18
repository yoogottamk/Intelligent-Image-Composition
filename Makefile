build:
	docker build -t friendblend src

run:
	docker run --rm -it -v ${PWD}/images:/images friendblend
