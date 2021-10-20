SOURCES = main.cpp
HEADERS = 
TARGET = run

$(TARGET): $(SOURCES) $(HEADERS)
	dpcpp -fsycl $(SOURCES) $(HEADERS) -o $(TARGET)
