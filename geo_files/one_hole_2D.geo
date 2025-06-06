h = 0.10;
//+
Point(1) = {0, 0, 0, h};
//+
Point(2) = {1, 0, 0, h};
//+
Line(1) = {1, 2};
//+

//+
Point(9) = {0.4, 0.4, 0, h};
//+
Point(10) = {0.6, 0.4, 0, h};
//+
Line(9) = {9, 10};
//+
//+
Extrude {0, 0.2, 0} {
  Point{9}; Point{10}; 
}
//+
Line(14) = {12, 11};

//+
Extrude {0, 1, 0} {
  Point{1}; Point{2}; 
}

Line(17) = {13, 14};//+
//+
Curve Loop(1) = {17, -16, -1, 15};
//+
Curve Loop(2) = {14, -10, 9, 11};

Plane Surface(1) = {1, 2};
//+
Physical Surface(19) = {1};

Physical Curve("1", 20) = {16, 1};
//+
//+
