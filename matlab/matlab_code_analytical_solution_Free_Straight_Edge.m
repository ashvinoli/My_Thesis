%defining variables in kN, m
%code is for model number 15
clear, clc
syms alp phi x
chord = 5;
a=3; %radius of cylinder
l=10; %length of cylinder
h=0.05; %thickness of cylinder
nu=0.2; %poisson's ratio
csq=h^2/(12*a^2); %a parameter
E=25000000; %modulus of elasticity
p=10; %udl in gravity direction
%ang=atan(chord/(2*a))*2; %central angle
ang = pi/2;
n=30; %no of iteration
A=sym('A%d%d',[n,4]); %matrix of arbitrary constants
k1=E*h/(1-nu^2);
k2=E*h^3/(12*(1-nu^2));

f = sym('f%d%d',[1,n]);
lam = sym('lam%d%d',[1,n]);
y1 = sym('y1%d%d',[1,n]);
y2 = sym('y2%d%d',[1,n]);
b1 = sym('b1%d%d',[1,n]);
b2= sym('b2%d%d',[1,n]);
solA0 = sym('SA%d%d',[1,n]);
solB0 = sym('SB%d%d',[1,n]);
solC0 = sym('SC%d%d',[1,n]);
C = sym('C%d%d',[n,4]);

for m=1:n
f(m)=sym(0);
lam(m)=m*pi*a/l; %lambda a parameter
l2=-(4*lam(m)^2-2);
l3=6*lam(m)^4+csq*(1-nu^2)*lam(m)^4-8*lam(m)^2+1;
l4=-(4*(1+csq)*lam(m)^6-(8-2*nu^2)*lam(m)^4+4*lam(m)^2);
l5=(1+4*csq)*lam(m)^8+(1-nu^2)*(4+1/csq)*lam(m)^4;
eqn=alp^8+l2*alp^6+l3*alp^4+l4*alp^2+l5==0;
sol=solve(eqn,alp);
y1(m)=abs(real(sol(1)));
y2(m)=abs(real(sol(3)));
b1(m)=abs(imag(sol(1)));
b2(m)=abs(imag(sol(3)));
f(m)=A(m,1)*cos(b1(m)*phi)*cosh(y1(m)*phi) +A(m,2)*sin(b1(m)*phi)*sinh(y1(m)*phi)+A(m,3)*cos(b2(m)*phi)*cosh(y2(m)*phi)+A(m,4)*sin(b2(m)*phi)*sinh(y2(m)*phi);
end
%calculation of particular solution
syms A0 B0 C0
for m=1:n
if mod(m,2)==0
k=0;
else
k=1;
end
a1=lam(m)^2+(1-nu)/2;
a2=-lam(m)*0.5*(1+nu);
a3=nu*lam(m);
a4=(1+nu)*lam(m)/2;
a5=-((1-nu)*lam(m)^2/2+1+csq*(1+2*(1-nu)*lam(m)^2));
a6=1+csq*(1+(2-nu)*lam(m)^2);
a7=-4*a^2*p*k/(k1*m*pi);
a8=nu*lam(m);
a9=-(1+csq*(1+(2-nu)*lam(m)^2));
a10=1+csq*(lam(m)^4+2*lam(m)^2+1);
a11=4*k*p*a^2/(m*pi*k1);
eqn1=A0*a1+B0*a2+C0*a3==0;
eqn2= A0*a4+B0*a5+C0*a6==a7;
eqn3=A0*a8+B0*a9+C0*a10==a11;
soln=solve(eqn1,eqn2,eqn3,A0,B0,C0);
solA0(m)=vpa(soln.A0);
solB0(m)=vpa(soln.B0);
solC0(m)=vpa(soln.C0);
end
%calculation of arbitrary constants
for m=1:n
fm1=diff(f(m),phi);
fm2=diff(f(m),phi,2);
fm3=diff(f(m),phi,3);
fm4=diff(f(m),phi,4);
fm5=diff(f(m),phi,5);
ui=(-fm2*lam(m)-nu*lam(m)^3*f(m)+csq*(-(1+nu)*(2-nu)*lam(m)^3*fm2/(1-nu)+(1+nu)*lam(m)*fm4/(1-nu)-4*nu*lam(m)^3*f(m)+2*nu*lam(m)*fm2/(1-nu))+solA0(m)*cos(phi));
vi=(-(2+nu)*lam(m)^2*fm1+fm3-csq*(2*(2-nu)*lam(m)^4*fm1/(1-nu)-(4-3*nu+nu^2)*lam(m)^2*fm3/(1-nu)+fm5)+solB0(m)*sin(phi));
wi=(lam(m)^4*f(m)-2*lam(m)^2*fm2+fm4+csq*(4*lam(m)^4*f(m)-2*(2-2*nu+nu^2)*lam(m)^2*fm2/(1-nu)+fm4)+solC0(m)*cos(phi));
dux=diff(ui,x);
dup=diff(ui,phi);
dvx=diff(vi,x);
dvp=diff(vi,phi);
dwx2=diff(wi,x,2);
dwp2=diff(wi,phi,2);
dwxp=diff(diff(wi,x),phi);
N2=k1*((dvp-wi)/a+nu*dux);
N12=E*h*(dvx+dup/a)/(2*(1+nu));
M2=-k2*(nu*dwx2+(dvp+dwp2)/a^2);
M12=k2*(1-nu)*(dvx+dwxp)/a;
M2phi = diff(M2,phi);
M21x = diff(M12,x);
eqn4=subs(N2,phi,ang/2)==0;
eqn5=subs(M2,phi,ang/2)==0;
eqn6=subs(N12-M12/a,phi,ang/2)==0;
eqn7=subs((1/a)*M2phi+2*M21x,phi,ang/2)==0;
solna=vpasolve(eqn4,eqn5,eqn6,eqn7,A(m,1),A(m,2),A(m,3),A(m,4));
C(m,1)=solna.(sprintf('A%d1',m));
C(m,2)=solna.(sprintf('A%d2',m));
C(m,3)=solna.(sprintf('A%d3',m));
C(m,4)=solna.(sprintf('A%d4',m));
end
%calculation of displacements
u=0;
v=0;
w=0;
for m=1:n
f(m)=C(m,1)*cos(b1(m)*phi)*cosh(y1(m)*phi)+C(m,2)*sin(b1(m)*phi)*sinh(y1(m)*phi)+C(m,3)*cos(b2(m)*phi)*cosh(y2(m)*phi)+C(m,4)*sin(b2(m)*phi)*sinh(y2(m)*phi);
fm1=diff(f(m),phi);
fm2=diff(f(m),phi,2);
fm3=diff(f(m),phi,3);
fm4=diff(f(m),phi,4);
fm5=diff(f(m),phi,5);
u=u+(-fm2*lam(m)-nu*lam(m)^3*f(m)+csq*(-(1+nu)*(2-nu)*lam(m)^3*fm2/(1-nu)+(1+nu)*lam(m)*fm4/(1-nu)-4*nu*lam(m)^3*f(m)+2*nu*lam(m)*fm2/(1-nu))+solA0(m)*cos(phi))*cos(lam(m)*x/a);
v=v+(-(2+nu)*lam(m)^2*fm1+fm3-csq*(2*(2-nu)*lam(m)^4*fm1/(1-nu)-(4-3*nu+nu^2)*lam(m)^2*fm3/(1-nu)+fm5)+solB0(m)*sin(phi))*sin(lam(m)*x/a);
w=w+(lam(m)^4*f(m)-2*lam(m)^2*fm2+fm4+csq*(4*lam(m)^4*f(m)-2*(2-2*nu+nu^2)*lam(m)^2*fm2/(1-nu)+fm4)+solC0(m)*cos(phi))*sin(lam(m)*x/a);
end
%calculation of force components
dux=diff(u,x);
dup=diff(u,phi);
dvx=diff(v,x);
dvp=diff(v,phi);
dwx2=diff(w,x,2);
dwp2=diff(w,phi,2);
dwxp=diff(diff(w,x),phi);
N1=k1*(dux+nu*(dvp-w)/a);
N2=k1*((dvp-w)/a+nu*dux);
N12=E*h*(dvx+dup/a)/(2*(1+nu));
M1=-k2*(dwx2+nu*(dvp+dwp2)/a^2);
M2=-k2*(nu*dwx2+(dvp+dwp2)/a^2);
M12=k2*(1-nu)*(dvx+dwxp)/a;
Q2=-k2*(diff((dwp2/a^2+dwx2),phi))/a;