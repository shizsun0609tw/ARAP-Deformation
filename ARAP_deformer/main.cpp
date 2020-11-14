#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <vector>
#include <numeric>
#include <iostream>
#include <limits>
#include <Windows.h>
#include <gl/GL.h>
#include <glut.h>

#include <set>
#include <map>

#include "glm.h"
#include "mtxlib.h"
#include "trackball.h"

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseCore>
#include <Eigen/SparseCholesky>

using namespace std;

// ----------------------------------------------------------------------------------------------------
// global variables

_GLMmodel *mesh;
_GLMmodel *originMesh;

int WindWidth, WindHeight;
int last_x , last_y;
int select_x, select_y;

typedef enum { SELECT_MODE, DEFORM_MODE } ControlMode;
ControlMode current_mode = SELECT_MODE;

vector<float*> colors;
vector<vector<int> > handles;
int selected_handle_id = -1;
bool deform_mesh_flag = false;

float err = 1;
float err_limit = 0.01f;
map<int, set<int>> connectedMap;
vector<vector<vector3>> e, e_p;

// ----------------------------------------------------------------------------------------------------
// render related functions

void Reshape(int width, int height)
{
	int base = min(width , height);

	tbReshape(width, height);
	glViewport(0 , 0, width, height);
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(45.0,(GLdouble)width / (GLdouble)height , 1.0, 128.0);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(0.0, 0.0, -3.5);

	WindWidth = width;
	WindHeight = height;
}

void Display(void)
{
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glPushMatrix();
	tbMatrix();

	// render solid model
	glEnable(GL_LIGHTING);
	glColor3f(1.0 , 1.0 , 1.0f);
	glPolygonMode(GL_FRONT_AND_BACK , GL_FILL);
	glmDraw(mesh , GLM_SMOOTH);

	// render wire model
	glPolygonOffset(1.0 , 1.0);
	glEnable(GL_POLYGON_OFFSET_FILL);
	glLineWidth(1.0f);
	glColor3f(0.6 , 0.0 , 0.8);
	glPolygonMode(GL_FRONT_AND_BACK , GL_LINE);
	glmDraw(mesh , GLM_SMOOTH);

	// render handle points
	glPointSize(10.0);
	glEnable(GL_POINT_SMOOTH);
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	for(int handleIter=0; handleIter<handles.size(); handleIter++)
	{
		glColor3fv(colors[ handleIter%colors.size() ]);
		for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
		{
			int idx = handles[handleIter][vertIter];
			glVertex3fv((float *)&mesh->vertices[3 * idx]);
		}
	}
	glEnd();

	glPopMatrix();

	glFlush();  
	glutSwapBuffers();
}

// ----------------------------------------------------------------------------------------------------
// mouse related functions

vector3 Unprojection(vector2 _2Dpos)
{
	float Depth;
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	glReadPixels((int)_2Dpos.x , viewport[3] - (int)_2Dpos.y , 1, 1, GL_DEPTH_COMPONENT, GL_FLOAT, &Depth);

	double X = _2Dpos.x;
	double Y = _2Dpos.y;
	double wpos[3] = {0.0 , 0.0 , 0.0};

	gluUnProject(X , ((double)viewport[3] - Y) , (double)Depth , ModelViewMatrix , ProjectionMatrix , viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector3(wpos[0] , wpos[1] , wpos[2]);
}

vector2 projection_helper(vector3 _3Dpos)
{
	int viewport[4];
	double ModelViewMatrix[16];    // Model_view matrix
	double ProjectionMatrix[16];   // Projection matrix

	glPushMatrix();
	tbMatrix();

	glGetIntegerv(GL_VIEWPORT, viewport);
	glGetDoublev(GL_MODELVIEW_MATRIX, ModelViewMatrix);
	glGetDoublev(GL_PROJECTION_MATRIX, ProjectionMatrix);

	glPopMatrix();

	double wpos[3] = {0.0 , 0.0 , 0.0};
	gluProject(_3Dpos.x, _3Dpos.y, _3Dpos.z, ModelViewMatrix, ProjectionMatrix, viewport, &wpos[0] , &wpos[1] , &wpos[2]);

	return vector2(wpos[0], (double)viewport[3]-wpos[1]);
}


Eigen::MatrixXd ExeCholeskySolver(Eigen::SparseMatrix<double>* A, Eigen::SparseMatrix<double>* b)
{
	Eigen::SimplicialCholesky<Eigen::SparseMatrix<double>> solver(*A);

	cout << "---------------------------------------------------------" << endl;
	cout << "Start to solve LeastSquare method by Cholesky solver" << endl;
	cout << "---------------------------------------------------------" << endl;

	Eigen::MatrixXd res = solver.solve(*b);

	return res;
}

map<int, set<int>> GetConnectMap(_GLMmodel* originMesh)
{
	map<int, set<int>> res;

	for (int i = 0; i < originMesh->numtriangles; ++i)
	{
		for (int j = 0; j < 3; ++j)
		{
			res.insert(pair<int, set<int>>(originMesh->triangles[i].vindices[j], set<int>()));

			res[originMesh->triangles[i].vindices[j]].insert(originMesh->triangles[i].vindices[(j + 1) % 3]);
			res[originMesh->triangles[i].vindices[j]].insert(originMesh->triangles[i].vindices[(j + 2) % 3]);
		}
	}

	return res;
}

vector<double> LeastSquareSolver(set<int> controlIndices, map<int, set<int>> connectedMap, vector<Eigen::Vector3f> b_top,_GLMmodel* originMesh)
{
	// Solve Ax = b
	Eigen::SparseMatrix<double> A(originMesh->numvertices + controlIndices.size(), originMesh->numvertices);
	Eigen::SparseMatrix<double> b(originMesh->numvertices + controlIndices.size(), 3);
	vector<Eigen::Triplet<double>> tripletListA, tripletListb;
	Eigen::Triplet<double> value;
	
	// Laplacian
	int row = 0;
	for (auto iterMap = ++connectedMap.begin(); iterMap != connectedMap.end(); ++iterMap, ++row)
	{
		int col = row;

		tripletListA.push_back(Eigen::Triplet<double>(row, col, (*iterMap).second.size()));
		for (auto iterSet = (*iterMap).second.begin(); iterSet != (*iterMap).second.end(); ++iterSet)
		{
			col = (*iterSet) - 1;

			value = Eigen::Triplet<double>(row, col, -1.0f);
			tripletListA.push_back(value);
		}
	}
	
	// b_top (0.5 * (Ri + Rj)(eij))
	for (int i = 0; i < originMesh->numvertices; ++i)
	{
		value = Eigen::Triplet<double>(i, 0, b_top[i][0]);
		tripletListb.push_back(value);
		value = Eigen::Triplet<double>(i, 1, b_top[i][1]);
		tripletListb.push_back(value);
		value = Eigen::Triplet<double>(i, 2, b_top[i][2]);
		tripletListb.push_back(value);
	}
	
	// Constraint
	row = 0;
	for (auto iter = controlIndices.begin(); iter != controlIndices.end(); ++iter, ++row)
	{
		value = Eigen::Triplet<double>(mesh->numvertices + row, *iter - 1, 1);
		tripletListA.push_back(value);

		for (int i = 0; i < 3; ++i)
		{
			value = Eigen::Triplet<double>(mesh->numvertices + row, i, mesh->vertices[*iter * 3 + i]);
			tripletListb.push_back(value);
		}
	}
	
	A.setFromTriplets(tripletListA.begin(), tripletListA.end());
	b.setFromTriplets(tripletListb.begin(), tripletListb.end());

	Eigen::SparseMatrix<double> ATA = A.transpose() * A;
	b = A.transpose() * b;

	Eigen::MatrixXd x;

	x = ExeCholeskySolver(&ATA, &b);

	vector<double> res;

	for (int i = 0; i < x.rows(); ++i)
	{
		for (int j = 0; j < x.cols(); ++j)
		{
			res.push_back(x.row(i).col(j).value());
		}
	}

	return res;
}

vector<vector<vector3>> CalE(_GLMmodel* iMesh)
{
	vector<vector<vector3>> res;

	for (int i = 0; i < iMesh->numvertices; ++i)
	{
		res.push_back(vector<vector3>());
		for (auto iter = connectedMap[i].begin(); iter != connectedMap[i].end(); ++iter)
		{
			vector3 v = (vector3(
				iMesh->vertices[3 * (i + 1) + 0] - iMesh->vertices[3 * (*iter) + 0],
				iMesh->vertices[3 * (i + 1) + 1] - iMesh->vertices[3 * (*iter) + 1],
				iMesh->vertices[3 * (i + 1) + 2] - iMesh->vertices[3 * (*iter) + 2]));
			res[i].push_back(v);
		}
	}

	return res;
}

Eigen::Matrix3f CalR(int index)
{
	Eigen::Matrix3f R;
	Eigen::MatrixXf eij(1, 3), eij_p(1, 3);
	Eigen::Matrix3f Si = Eigen::Matrix3f::Zero();

	for (int j = 0; j < connectedMap[index].size(); ++j)
	{
		for (int x = 0; x < 3; ++x)
		{
			for (int y = 0; y < 3; ++y)
			{
				Si(x, y) += e[index][j][x] * e_p[index][j][y];
			}
		}
	}

	Eigen::JacobiSVD<Eigen::MatrixXf> svd(Si, Eigen::ComputeFullU | Eigen::ComputeFullV);
	Eigen::MatrixXf U = svd.matrixU();
	Eigen::MatrixXf V = svd.matrixV();
	Eigen::MatrixXf S = svd.singularValues();

	R = V * U.transpose();

	return R;
}

_GLMmodel* ReconstructModel(_GLMmodel* recMesh, vector<double> solveVertices)
{
	for (int i = 0; i < solveVertices.size(); ++i)
	{
		recMesh->vertices[i + 3] = solveVertices[i];
	}

	return recMesh;
}

void CalError(vector<double> solveVertices)
{
	float err_tmp = 0;

	for (int i = 0; i < solveVertices.size(); ++i)
	{
		err_tmp = max(fabs(mesh->vertices[i + 3] - solveVertices[i]), err_tmp);
	}

	err = min(err_tmp, err);
}

vector<double> DeformationIteration()
{
	vector<double> res;
	
	e_p = CalE(mesh);

	vector<Eigen::Vector3f> b_top;
	Eigen::Matrix3f Ri, Rj;
	Eigen::Vector3f eij, b_temp;
	
	for (int i = 0; i < mesh->numvertices; ++i)
	{
		b_temp = Eigen::Vector3f::Zero();
		for (auto iter = connectedMap[i + 1].begin(); iter != connectedMap[i + 1].end(); ++iter)
		{
			eij = Eigen::Vector3f(originMesh->vertices[3 * (i + 1) + 0] - originMesh->vertices[3 * (*iter) + 0],
								  originMesh->vertices[3 * (i + 1) + 1] - originMesh->vertices[3 * (*iter) + 1],
								  originMesh->vertices[3 * (i + 1) + 2] - originMesh->vertices[3 * (*iter) + 2]);

			Ri = CalR(i);
			Rj = CalR(*iter - 1);
			b_temp += 0.5f * (Ri + Rj) * eij;
		}
		b_top.push_back(b_temp);
	}
	
	set<int> controlIndices;

	for (int i = 0; i < handles.size(); ++i)
	{
		for (int j = 0; j < handles[i].size(); ++j)
		{
			controlIndices.insert(handles[i][j]);
		}
	}

	res = LeastSquareSolver(controlIndices, connectedMap, b_top, originMesh);

	return res;
}

void mouse(int button, int state, int x, int y)
{
	tbMouse(button, state, x, y);

	if( current_mode==SELECT_MODE && button==GLUT_RIGHT_BUTTON )
	{
		if(state==GLUT_DOWN)
		{
			select_x = x;
			select_y = y;
		}
		else
		{
			vector<int> this_handle;

			// project all mesh vertices to current viewport
			for(int vertIter=0; vertIter<mesh->numvertices; vertIter++)
			{
				vector3 pt(mesh->vertices[3 * vertIter + 0] , mesh->vertices[3 * vertIter + 1] , mesh->vertices[3 * vertIter + 2]);
				vector2 pos = projection_helper(pt);

				// if the projection is inside the box specified by mouse click&drag, add it to current handle
				if(pos.x>=select_x && pos.y>=select_y && pos.x<=x && pos.y<=y)
				{
					this_handle.push_back(vertIter);
				}
			}
			if (this_handle.size() != 0)
			{
				handles.push_back(this_handle);
			}
		}
	}
	// select handle
	else if( current_mode==DEFORM_MODE && button==GLUT_RIGHT_BUTTON && state==GLUT_DOWN )
	{
		// project all handle vertices to current viewport
		// see which is closest to selection point
		double min_dist = 999999;
		int handle_id = -1;
		for(int handleIter=0; handleIter<handles.size(); handleIter++)
		{
			for(int vertIter=0; vertIter<handles[handleIter].size(); vertIter++)
			{
				int idx = handles[handleIter][vertIter];
				vector3 pt(mesh->vertices[3 * idx + 0] , mesh->vertices[3 * idx + 1] , mesh->vertices[3 * idx + 2]);
				vector2 pos = projection_helper(pt);

				double this_dist = sqrt((double)(pos.x-x)*(pos.x-x) + (double)(pos.y-y)*(pos.y-y));
				if(this_dist<min_dist)
				{
					min_dist = this_dist;
					handle_id = handleIter;
				}
			}
		}

		selected_handle_id = handle_id;
		deform_mesh_flag = true;
	}
	else if (current_mode == DEFORM_MODE && button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
	{
		err = INT_MAX;
		int iteration = 0;
		while (err > err_limit)
		{
			iteration++;
			vector<double> res = DeformationIteration();
			CalError(res);
			mesh = ReconstructModel(mesh, res);
			printf("iteration:%d, err:%f\n\n", iteration, err);

			Display();
		}
		printf("*********************************************\n");
		printf("***                finished               ***\n");
		printf("*********************************************\n\n");
	}

	if(button == GLUT_RIGHT_BUTTON && state == GLUT_UP)
		deform_mesh_flag = false;

	last_x = x;
	last_y = y;
}

void motion(int x, int y)
{
	tbMotion(x, y);

	// if in deform mode and a handle is selected, deform the mesh
	if( current_mode==DEFORM_MODE && deform_mesh_flag==true )
	{
		matrix44 m;
		vector4 vec = vector4((float)(x - last_x) / 1000.0f , (float)(y - last_y) / 1000.0f , 0.0 , 1.0);
		
		gettbMatrix((float *)&m);
		vec = m * vec;

		// deform handle points
		for(int vertIter=0; vertIter<handles[selected_handle_id].size(); vertIter++)
		{
			int idx = handles[selected_handle_id][vertIter];
			vector3 pt(mesh->vertices[3*idx+0]+vec.x, mesh->vertices[3*idx+1]+vec.y, mesh->vertices[3*idx+2]+vec.z);
			mesh->vertices[3 * idx + 0] = pt[0];
			mesh->vertices[3 * idx + 1] = pt[1];
			mesh->vertices[3 * idx + 2] = pt[2];
		}
	}

	last_x = x;
	last_y = y;
}

// ----------------------------------------------------------------------------------------------------
// keyboard related functions

void keyboard(unsigned char key, int x, int y )
{
	switch(key)
	{
	case 'd':
		current_mode = DEFORM_MODE;
		break;
	default:
	case 's':
		current_mode = SELECT_MODE;
		break;
	}
}

// ----------------------------------------------------------------------------------------------------
// main function

void timf(int value)
{
	glutPostRedisplay();
	glutTimerFunc(1, timf, 0);
}

void Init() 
{
	connectedMap = GetConnectMap(originMesh);
	e = CalE(originMesh);
}

int main(int argc, char *argv[])
{
	// compute SVD decomposition of a matrix m
	// SVD: m = U * S * V^T
	Eigen::MatrixXf m = Eigen::MatrixXf::Random(3,3);
	cout << "Here is the matrix m:" << endl << m << endl;
	Eigen::JacobiSVD<Eigen::MatrixXf> svd(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
	const Eigen::Matrix3f U = svd.matrixU();
	// note that this is actually V^T!!
	const Eigen::Matrix3f V = svd.matrixV();
	const Eigen::VectorXf S = svd.singularValues();

	WindWidth = 800;
	WindHeight = 800;

	GLfloat light_ambient[] = {0.0, 0.0, 0.0, 1.0};
	GLfloat light_diffuse[] = {0.8, 0.8, 0.8, 1.0};
	GLfloat light_specular[] = {1.0, 1.0, 1.0, 1.0};
	GLfloat light_position[] = {0.0, 0.0, 1.0, 0.0};

	// color list for rendering handles
	float red[] = {1.0, 0.0, 0.0};
	colors.push_back(red);
	float yellow[] = {1.0, 1.0, 0.0};
	colors.push_back(yellow);
	float blue[] = {0.0, 1.0, 1.0};
	colors.push_back(blue);
	float green[] = {0.0, 1.0, 0.0};
	colors.push_back(green);

	glutInit(&argc, argv);
	glutInitWindowSize(WindWidth, WindHeight);
	glutInitDisplayMode(GLUT_RGB | GLUT_DEPTH | GLUT_DOUBLE);
	glutCreateWindow("ARAP");

	glutReshapeFunc(Reshape);
	glutDisplayFunc(Display);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutKeyboardFunc(keyboard);
	glClearColor(0, 0, 0, 0);

	glLightfv(GL_LIGHT0, GL_AMBIENT, light_ambient);
	glLightfv(GL_LIGHT0, GL_DIFFUSE, light_diffuse);
	glLightfv(GL_LIGHT0, GL_SPECULAR, light_specular);
	glLightfv(GL_LIGHT0, GL_POSITION, light_position);

	glEnable(GL_LIGHT0);
	glDepthFunc(GL_LESS);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	tbInit(GLUT_LEFT_BUTTON);
	tbAnimate(GL_TRUE);

	glutTimerFunc(40, timf, 0); // Set up timer for 40ms, about 25 fps

	// load 3D model
	mesh = glmReadOBJ("../data/man.obj");
	originMesh = glmReadOBJ("../data/man.obj");

	glmUnitize(mesh);
	glmFacetNormals(mesh);
	glmVertexNormals(mesh , 90.0);

	glmUnitize(originMesh);
	glmFacetNormals(originMesh);
	glmVertexNormals(originMesh, 90.0);

	Init();

	glutMainLoop();

	return 0;

}