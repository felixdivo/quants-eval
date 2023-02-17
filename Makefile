all: build run

build:
	echo "Build image"
	export COMPOSE_PROJ_NAME=timeseries_vit	
	docker compose -f .docker/compose.yml rm timeseries_vit
	docker compose -f .docker/compose.yml build timeseries_vit
	echo "Finished build image"

run:
	bash make_scripts/run.sh

stop:
	docker compose -f .docker/compose.yml stop timeseries_vit

remove:
	docker compose -f .docker/compose.yml rm timeseries_vit

stop_remove: stop remove

