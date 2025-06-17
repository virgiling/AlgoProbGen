.PHONY: clean build

build: clean
	@.venv/bin/python src/generator.py
	@zip -r Problem.zip data/
	@mv src/generator.py src/$(shell date +%y_%m_%d).py

clean:
	@rm -rf data/ Problem.zip
