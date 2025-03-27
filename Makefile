### General

.PHONY: help parquet publish devserver pmtiles init

all: help

help:		## output help for all targets
	@echo "These are the targets of this Makefile:"
	@echo
	@awk 'BEGIN {FS = ":.*?## "}; \
		/^###/ {printf "\n\033[1;33m%s\033[0m\n", substr($$0, 5)}; \
		/^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-18s\033[0m %s\n", $$1, $$2}' \
		$(MAKEFILE_LIST)

# Check if conda is initialized 
ifeq ($(shell test -z "$$INIT_SH_LOADED" && echo 1),1)
$(error Please run "source init.sh" before running make)
endif

PARQUET_FILES = assets/swissBUILDINGS3D_3-0_1112-13_Building_solid/chunk_00.parquet\
	        assets/swissBUILDINGS3D_3_0_Building_solid/chunk_00.parquet \
		assets/swissBUILDINGS3D_3-0_1112-13_Roof_solid/chunk_00.parquet \
		assets/swissBUILDINGS3D_3_0_Roof_solid/chunk_00.parquet

PARQUET_FILES_2D := $(patsubst %/chunk_00.parquet,%_2d/chunk_00.parquet,$(PARQUET_FILES))


SWISSTLM3D_URL = https://data.geo.admin.ch/ch.swisstopo.swisstlm3d/swisstlm3d_2024-03/swisstlm3d_2024-03_2056_5728.gpkg.zip
SWISSTLM3D_BASENAME = $(notdir $(SWISSTLM3D_URL))
# adjust the absolute path of SWISSTLM3D_FILENAME to your liking
SWISSTLM3D_FILENAME = $(HOME)/data/swisstopo/$(SWISSTLM3D_BASENAME)

### ETL

dependency: assets/dependency_processing.png ## Show the dependency graph of this Makefile

assets/dependency_processing.png: Makefile 	
	uvx makefile2dot --direction LR | dot -Tpng > $@

download:	assets/data.sqlite assets/swissBUILDINGS3D_3_0.gdb/gdb assets/swissBUILDINGS3D_3-0_1112-13.gdb/gdb assets/data.sqlite $(SWISSTLM3D_FILENAME) ## download and unzip buildings, address data and TLM data (may take a while, large data)

publish:	assets/swissBUILDINGS3D_3-0_1112-13_Building_solid_2d/chunk_00.parquet assets/swissBUILDINGS3D_3-0_1112-13_Roof_solid_2d/chunk_00.parquet assets/swissBUILDINGS3D_3_0_Building_solid_2d/chunk_00.parquet  assets/swissBUILDINGS3D_3_0_Roof_solid_2d/chunk_00.parquet  assets/railway_bridges.parquet assets/road_bridges.parquet ## publish parquet files to the data dir
	rsync -av --progress $(dir $(word 1, $^)) ping13@s022.cyon.net:~/public_html/ping13.net/data
	rsync -av --progress $(dir $(word 2, $^)) ping13@s022.cyon.net:~/public_html/ping13.net/data
	rsync -av --progress $(dir $(word 3, $^)) ping13@s022.cyon.net:~/public_html/ping13.net/data
	rsync -av --progress $(dir $(word 4, $^)) ping13@s022.cyon.net:~/public_html/ping13.net/data
	rsync -av --progress $(word 5, $^) ping13@s022.cyon.net:~/public_html/ping13.net/data
	rsync -av --progress $(word 6, $^) ping13@s022.cyon.net:~/public_html/ping13.net/data

# this downloads the latest address database and unzips to have access to the SQLite databae
assets/data.sqlite:
	echo "Downloading Address data set"
	mkdir -p assets
	curl -J --output assets/data.sqlite "https://public.madd.bfs.admin.ch/data_ch.sqlite"

assets/swissBUILDINGS3D_3_0.gdb.zip:
	echo "Downloading Swiss Buildings"
	curl -J --output $@ "https://data.geo.admin.ch/ch.swisstopo.swissbuildings3d_3_0/swissbuildings3d_3_0_2024/swissbuildings3d_3_0_2024_2056_5728.gdb.zip"

assets/swissBUILDINGS3D_3_0.gdb/gdb: assets/swissBUILDINGS3D_3_0.gdb.zip
	(cd assets; unzip swissBUILDINGS3D_3_0.gdb.zip && cd .. && touch $@ )

assets/swissBUILDINGS3D_3-0_1112-13.gdb.zip:
	echo "Downloading subset of Swiss Buildings"
	curl -J --output $@ "https://data.geo.admin.ch/ch.swisstopo.swissbuildings3d_3_0/swissbuildings3d_3_0_2019_1112-13/swissbuildings3d_3_0_2019_1112-13_2056_5728.gdb.zip"

assets/swissBUILDINGS3D_3-0_1112-13.gdb/gdb: assets/swissBUILDINGS3D_3-0_1112-13.gdb.zip
	(cd assets; unzip -o swissBUILDINGS3D_3-0_1112-13.gdb.zip && cd .. && touch $@ )

$(SWISSTLM3D_FILE):
	echo "Downloading SwissTLM3D, beware, this is huge"
	mkdir -p `dirname $@`
	curl -o "$@" "$(SWISSTLM3D_URL)"


### ETL Addresses

create_addresses: web/public/addresses_full.parquet web/public/addresses.parquet	## create the address parquet files

web/public/addresses_full.parquet: scripts/addresses_sqlite2pq.py assets/data.sqlite
	mkdir -p web/public
	time python scripts/addresses_sqlite2pq.py --full --sqlite-file assets/data.sqlite $@ 

web/public/addresses.parquet: scripts/addresses_sqlite2pq.py assets/data.sqlite
	mkdir -p web/public
	time python scripts/addresses_sqlite2pq.py --sqlite-file assets/data.sqlite $@ 

### ETL Buildings

init: 				## initialize Python environment (conda)
	source init.sh && mamba activate myenv

create_buildings: $(PARQUET_FILES) $(PARQUET_FILES_2D)	## create building parquet files

pmtiles: web/public/roofs.pmtiles web/public/buildings.pmtiles 	## calculate building pmtiles for the web

%_Roof_solid/chunk_00.parquet: %.gdb/gdb scripts/swissbuildings3D_gdb2pq.py
	time python scripts/swissbuildings3D_gdb2pq.py $(patsubst %.gdb/gdb,%.gdb,$<) Roof_solid --chunk 100000

%_Roof_solid_2d/chunk_00.parquet: %_Roof_solid/chunk_00.parquet scripts/swissbuildings3D_process.py
	@echo "Creating $@"
	time python scripts/swissbuildings3D_process.py $(patsubst %/chunk_00.parquet,%,$<)

%_Building_solid/chunk_00.parquet: %.gdb/gdb scripts/swissbuildings3D_gdb2pq.py
	time python scripts/swissbuildings3D_gdb2pq.py $(patsubst %.gdb/gdb,%.gdb,$<) Building_solid --chunk 100000

%_Building_solid_2d/chunk_00.parquet: %_Building_solid/chunk_00.parquet scripts/swissbuildings3D_process.py
	@echo "Creating $@"
	python scripts/swissbuildings3D_process.py $(patsubst %/chunk_00.parquet,%,$<)

web/public/roofs.pmtiles: assets/swissBUILDINGS3D_3_0_Roof_solid_2d/chunk_00.parquet assets/railway_bridges.parquet assets/road_bridges.parquet scripts/pq2pmtiles.sh
	mkdir -p web/public/
	time bash scripts/pq2pmtiles.sh $(word 1, $^) $(word 2, $^) $(word 3, $^) $@

web/public/buildings.pmtiles: assets/swissBUILDINGS3D_3_0_Building_solid_2d/chunk_00.parquet assets/railway_bridges.parquet assets/road_bridges.parquet scripts/pq2pmtiles.sh
	mkdir -p web/public/
	time bash scripts/pq2pmtiles.sh $(word 1, $^) $(word 2, $^) $(word 3, $^) $@

### ETL Road and Railway Bridges - TLM

assets/bridge_parameters.db: scripts/create_bridge_db.sh
	bash scripts/create_bridge_db.sh $@

create_bridges: assets/road_bridges.parquet assets/railway_bridges.parquet ## create parametrized 3D bridges

assets/railway_bridges.parquet: assets/railway_bridges.gpkg.zip scripts/create_bridges_3d.py assets/bridge_parameters.db
	python scripts/create_bridges_3d.py $< $@

assets/road_bridges.parquet: assets/road_bridges.gpkg.zip scripts/create_bridges_3d.py assets/bridge_parameters.db
	python scripts/create_bridges_3d.py $< $@

assets/road_bridges.gpkg.zip: $(SWISSTLM3D_FILENAME)
	echo "creating road bridges, will take some time"
	mkdir -p `dirname $@`
	ogr2ogr -f GPKG $@ $(SWISSTLM3D_FILENAME) tlm_strassen_strasse \
	  -where "kunstbaute = 'Bruecke' OR kunstbaute = 'Bruecke mit Galerie' OR kunstbaute = 'Gedeckte Bruecke'"

assets/railway_bridges.gpkg.zip: $(SWISSTLM3D_FILENAME)
	echo "creating railway bridges, will take some time"
	mkdir -p `dirname $@`
	ogr2ogr -f GPKG $@ $(SWISSTLM3D_FILENAME) tlm_oev_eisenbahn -where "(kunstbaute = 'Bruecke' OR kunstbaute = 'Bruecke mit Galerie' OR kunstbaute = 'Gedeckte Bruecke') AND anzahl_spuren > 0 AND verkehrsmittel = 'Bahn'"


### Web
devserver: 	   ## run the map server with PMTiles locally
	cd web && npx vite --host

build_web:     ## build the web app for production
	cd web && npm run build --clean

test_web: build_web ## test the production build locally
	cd web && npx vite preview --host

publish_web: build_web ## publish the web app to my personal webserver
	rsync -av --progress web/dist/ ping13@s022.cyon.net:~/public_html/ping13.net/swiss-rooftop-explorer

