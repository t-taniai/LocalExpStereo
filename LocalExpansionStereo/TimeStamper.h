#pragma once
#include <time.h>

class TimeStamper
{
	clock_t previous;
	clock_t totalTime;
	bool _isTicking;

public:

	TimeStamper()
	{
		_isTicking = false;
		totalTime = 0;
	}

	~TimeStamper(void)
	{
	}
	
	void start()
	{
		if (_isTicking == false){
			_isTicking = true;
			previous = clock();
		}
	}

	void stop()
	{
		if (_isTicking){
			totalTime += clock() - previous;
			_isTicking = false;
		}
	}

	bool isTicking()
	{
		return _isTicking;
	}

	double getCurrentTime()
	{
		clock_t t = totalTime;
		if (_isTicking){
			t += clock() - previous;
		}
		return (double)t / CLOCKS_PER_SEC;
	}


};
