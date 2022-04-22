// VariablesAndTypes.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include <iostream>

using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	int i1 = 1;
	cout << "i1= " << i1 << endl;
	int i2;
	i2 = 2;
	cout << "i2= " << i2 << endl;
	int i3(3);
	cout << "i3= " << i3 << endl;

	double d1 = 2.2;
	double d2 = i1;
	int i4 = d1;
	cout << "d1= " << d1 << endl;
	cout << "d2= " << d2 << endl;
	cout << "i4= " << i4 << endl;

	char c1 = 'a';
	//char c2 = "b";
	cout << "c1= " << c1 << endl;
	// cout << "c2= " << c2 << endl;

	bool flag = false;
	cout << "flag= " << flag << endl;
	flag = i1;
	cout << "flag= " << flag << endl;
	flag = d1;
	cout << "flag= " << flag << endl;






	return 0;
}

