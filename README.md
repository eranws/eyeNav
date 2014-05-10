###Install

download and install opencv 2.4.8 or higher (tested on 2.4.8)

    http://opencv.org/downloads.html

install python packages (using:

    pip install httplib2
    pip install tornado

###Run

1. run websocket server:

	    cd server
        python chatdemo.py --port=8889

2. In your browser, open: (tested in chrome):

    	localhost:8889

3. (optional) test if websocket server gets events

    	python testsend.py

4. launch eyeNav:

    	python mosse.py