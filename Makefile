all: matrix

matrix: raw
	mkdir matrix
	cat raw/training.en | python text_to_vec.py -s matrix/en.span
	cat raw/training.ch | python text_to_vec.py -s matrix/ch.span

clean:
	rm -rf matrix
