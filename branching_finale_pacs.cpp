#include "particles.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <chrono> 
#include <map>
#include <string>
#include <filesystem>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/copy.h>
#include <thrust/fill.h>

using qgt = quadgrid_t<vector_t<real_t>>;
using idx_t = qgt::idx_t;

int main(){    
    try{
        std::cout << "Inizializzazione in corso...\n";      
        std::string out_dir = "parallel_branching_out";

        if (std::filesystem::exists(out_dir)) {
            std::filesystem::remove_all(out_dir);
        }

        std::filesystem::create_directories(out_dir);   
        std::ofstream en_file(out_dir + "/energies.csv");

        if (!en_file.is_open()) {
            std::cerr << "Errore: impossibile aprire energies.csv in scrittura!\n";
            return 1;
        }

        en_file << "step,t_us,E_cin,E_ela,E_frac,W_ext,E_bal,a_frac,v_tip_ms,n_rotte,frac_rateo\n";
        en_file << std::scientific << std::setprecision(10);            
            
        // Inizializzazione parametri fisici  
        const double E=32000.0;
        const double nu=0.2;
        const double rho=2.45e-9;
        const double Gc=0.003;
        const double mu_lame=E/(2.0*(1+nu));
        const double lambda_lame=(E*nu)/((1.0+nu)*(1.0-2.0*nu));

        // Inizializzazione parametri numerici
        const double h=0.125;
        const double alpha_flip=0.99;
        const double sigma_ext=1.0;
        double Lx=40.0;
        double Ly=100.0;
        const double hx = h, hy = h;
            
        const int n_celle_x=static_cast<int>(std::lround(Lx/hx));
        const int n_celle_y=static_cast<int>(std::lround(Ly/hy));

        Lx = n_celle_x * hx;
        Ly = n_celle_y * hy;

        const double k=0.0;
        const double l0=2*h;

        // Celle di margine
        const int margine=4;
        const int n_cols=n_celle_x+margine*2;
        const int n_rows=n_celle_y+margine*2;
        const double altezza_margine=margine*hy;
        const double larghezza_margine=margine*hx;

        // definizione parametri temporali
        const double dt=0.025e-6;
        const double t_tot=80e-6;
        const int num_steps=static_cast<int>(std::ceil(t_tot/dt));
            
        // Inizializzazione griglia
        qgt grid;
        grid.set_sizes(n_rows,n_cols,hx,hy);
        const idx_t nn=grid.num_global_nodes();
            
        const double n_part_cella=4;
        const idx_t num_particles_init=static_cast<idx_t> (n_celle_x)*static_cast<idx_t>(n_celle_y)*n_part_cella;

        if (qgt::gt (3, n_cols - 1, n_rows - 1, n_rows) != nn - 1) {
            std::cerr << "ERRORE indicizzazione\n";
            return 1;
        }

        // Inizializzazione posizione particelle (Gauss-Legendre)
        const double g1=0.5*(1.0-1.0/std::sqrt (3.0));
        const double g2=0.5*(1.0+1.0/std::sqrt(3.0));
        int count_x=0;
        int count_y=0;

        auto x_generator=[&count_x,n_celle_x,hx,larghezza_margine,g1,g2]()->double {
            const int cell_idx=count_x/4, local_idx=count_x%4;
            const int col=cell_idx%n_celle_x;
            ++count_x;
            if (local_idx%2==0)return larghezza_margine+col*hx+hx*g1;
            else return larghezza_margine+col*hx+hx*g2;
        };

        auto y_generator=[&count_y, n_celle_x,hy,altezza_margine,g1,g2]()->double {
            const int cell_idx=count_y/4, local_idx=count_y%4;
            const int row=cell_idx/n_celle_x;
            ++count_y;
            if (local_idx<2) return altezza_margine+row*hy+hy*g1;
            else return altezza_margine+row*hy+hy*g2;
        };

        particles_t particles(num_particles_init,{"id"},{"massa","area","vx","vy","px","py","Lxx","Lxy","Lyx","Lyy","minus_sxx","minus_syy","minus_sxy",
        "vpx","vpy","ax","ay","cp","gcx","gcy","eps_xx","eps_yy","eps_xy","sig_xx","sig_xy","sig_yy","damage","history"},grid,x_generator,y_generator);
        
        //inizializzazione variabili particelle
        std::iota(particles.iprops["id"].begin(),particles.iprops["id"].end(),0);

        const double area_p=hx*hy/4.0;
        const double massa_p=area_p*rho;
        
        //inizializzazione frattura iniziale
        const double history_iniz=1e3*Gc/(4*l0);
        const double frattura_x=larghezza_margine+0.5*Lx;
        const double fine_frattura_y=altezza_margine+0.5*Ly;
        const double inizio_frattura_y=altezza_margine+Ly;

        for (idx_t p=0;p<particles.num_particles;p++){
            particles.dprops["massa"][p]=massa_p;
            particles.dprops["area"][p]=area_p;
            particles.dprops["vx"][p]=0.0; particles.dprops["vy"][p]=0.0;
            particles.dprops["eps_xx"][p]=0.0; particles.dprops["eps_yy"][p]=0.0;
            particles.dprops["eps_xy"][p]=0.0; particles.dprops["sig_xx"][p]=0.0;
            particles.dprops["sig_yy"][p]=0.0; particles.dprops["sig_xy"][p]=0.0;
            particles.dprops["damage"][p]=0.0;
                
            bool in_notch = (particles.y[p]>=fine_frattura_y && particles.y[p]<=inizio_frattura_y && std::abs(frattura_x-particles.x[p])<=hx/2.0);
            if(in_notch) particles.dprops["history"][p]=history_iniz;
            else particles.dprops["history"][p]=0.0;
        }
            
        particles.memcpy_host_to_device();    

        // Inizializzazione variabili della griglia 
        std::map<std::string, device_vector_t<real_t>> grid_vars;
        grid_vars["gm"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gpx"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gpy"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gvx"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gvy"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gax"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gay"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gfx"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["gfy"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["g_c"]=device_vector_t<real_t>(nn,1.0);
        grid_vars["R_c"]=device_vector_t<real_t>(nn,0.0);
        grid_vars["D_c"]=device_vector_t<real_t>(nn,0.0);

        particles.device_grid_vars["gm"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gpx"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gpy"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gvx"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gvy"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gfx"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gfy"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gax"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["gay"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["R_c"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["D_c"]=device_vector_t<real_t>(nn, 0.0);
        particles.device_grid_vars["g_c"]=device_vector_t<real_t>(nn, 1.0);
            
        const double m_tol=1e-15 * massa_p; 

        // Definizione condizioni al bordo e imposizione sforzo
        std::vector<double> d_Fext(nn, 0.0);
        for(idx_t i=0;i<nn;i++){
            const idx_t r=i%(n_rows+1);
            const idx_t c=i/(n_rows+1);
            const double nx=c*hx;
            const double ny=r*hy;
            double trib=0.0;

            if((std::abs(nx-larghezza_margine)<1e-9) || (std::abs(nx-(larghezza_margine+Lx))<1e-9)){
                if((std::abs(ny-altezza_margine)<1e-9) || (std::abs(ny - (altezza_margine + Ly))<1e-9)){
                    trib=0.5;
                }
                else trib=1.0;
            }
                trib=trib*hx;
                if(std::abs(nx-larghezza_margine)<1e-9 && (ny>=altezza_margine) && (ny<=altezza_margine+Ly)) d_Fext[i]-=trib*sigma_ext;
                if(std::abs(nx-(larghezza_margine+Lx))<1e-9 && (ny>=altezza_margine) && (ny<=altezza_margine+Ly)) d_Fext[i]+=trib*sigma_ext;
        }
        
        thrust::device_vector<double> dev_Fext = d_Fext;
            
        // Variabili di controllo
        double W_ext=0.0;
        double fine_frattura_y_prev=fine_frattura_y;
        double t_prec=0.0;
        double v_tip_smoothed=0.0;

        auto t_start_print=std::chrono::high_resolution_clock::now();

        std::cout<<std::left 
            <<std::setw(8)<<"Step" 
            <<std::setw(12)<<"Tempo[us]" 
            <<std::setw(15)<<"Bilancio_E" 
            <<std::setw(15)<<"E_Frattura" 
            <<std::setw(12)<<"L_Cricca" 
            <<std::setw(12)<<"Vel_Punta" 
            <<std::setw(10)<<"T_Calc[s]"<<"\n"
            <<std::string(84, '-')<<"\n";

        //inizio ciclo temporale
        for(int step=0;step<num_steps;step++){
            //Fase 1: inizializzazione/reset griglia
            particles.update_ptcl_to_grd<particles_t::update_ptcl_to_grd_device>();

            for(auto const& k:{"gm", "gpx", "gpy", "gfx", "gfy", "gax", "gay", "R_c", "D_c"}) {
                thrust::fill(particles.device_grid_vars.at(k).begin(), particles.device_grid_vars.at(k).end(), 0.0);
            }
            
            //Fase 2: P2G di massa e quantità di moto
            const double* d_massa = thrust::raw_pointer_cast(particles.device_dprops.at("massa").data());
            const double* d_vx = thrust::raw_pointer_cast(particles.device_dprops.at("vx").data());
            const double* d_vy = thrust::raw_pointer_cast(particles.device_dprops.at("vy").data());
            double* d_px = thrust::raw_pointer_cast(particles.device_dprops.at("px").data());
            double* d_py = thrust::raw_pointer_cast(particles.device_dprops.at("py").data());

            thrust::counting_iterator<idx_t> first_p(0);
            thrust::counting_iterator<idx_t> last_p(particles.num_particles);
                
            thrust::for_each(thrust::device, first_p, last_p, 
                [d_massa, d_vx, d_vy, d_px, d_py] __device__ (idx_t p) {
                    d_px[p] = d_massa[p] * d_vx[p];
                    d_py[p] = d_massa[p] * d_vy[p];
                }
            );

            particles.p2g(grid_vars, {"massa", "px", "py"}, {"gm", "gpx", "gpy"}, false);
            
            double* d_gvx = thrust::raw_pointer_cast(particles.device_grid_vars.at("gvx").data());
            double* d_gvy = thrust::raw_pointer_cast(particles.device_grid_vars.at("gvy").data());
            const double* d_gm = thrust::raw_pointer_cast(particles.device_grid_vars.at("gm").data());
            const double* d_gpx = thrust::raw_pointer_cast(particles.device_grid_vars.at("gpx").data());
            const double* d_gpy = thrust::raw_pointer_cast(particles.device_grid_vars.at("gpy").data());
               
            thrust::counting_iterator<idx_t> first_n(0);
            thrust::counting_iterator<idx_t> last_n(nn);

            //calcolo velocità nodali
            thrust::for_each(thrust::device, first_n, last_n,
                [d_gvx, d_gvy, d_gm, d_gpx, d_gpy, m_tol] __device__ (idx_t n) {
                    if (d_gm[n] > m_tol) {
                        d_gvx[n] = d_gpx[n] / d_gm[n]; 
                        d_gvy[n] = d_gpy[n] / d_gm[n]; 
                    } else { 
                        d_gvx[n] = 0.0;
                        d_gvy[n] = 0.0; 
                    }
                } 
            );
            
            //Fase 3: G2PD di gradiente velocità e calcolo deformazioni
            //reset gradiente di deformazione
            thrust::fill(particles.device_dprops.at("Lxx").begin(), particles.device_dprops.at("Lxx").end(), 0.0);
            thrust::fill(particles.device_dprops.at("Lyy").begin(), particles.device_dprops.at("Lyy").end(), 0.0);
            thrust::fill(particles.device_dprops.at("Lxy").begin(), particles.device_dprops.at("Lxy").end(), 0.0);
            thrust::fill(particles.device_dprops.at("Lyx").begin(), particles.device_dprops.at("Lyx").end(), 0.0);

            particles.g2pd(grid_vars, {"gvx", "gvy"}, {"Lxx", "Lyx"}, {"Lxy", "Lyy"}, false);

            const double* d_Lxx = thrust::raw_pointer_cast(particles.device_dprops.at("Lxx").data());
            const double* d_Lyy = thrust::raw_pointer_cast(particles.device_dprops.at("Lyy").data());
            const double* d_Lxy = thrust::raw_pointer_cast(particles.device_dprops.at("Lxy").data());
            const double* d_Lyx = thrust::raw_pointer_cast(particles.device_dprops.at("Lyx").data());   
            
            double* d_eps_xx  = thrust::raw_pointer_cast(particles.device_dprops.at("eps_xx").data());
            double* d_eps_yy  = thrust::raw_pointer_cast(particles.device_dprops.at("eps_yy").data());
            double* d_eps_xy  = thrust::raw_pointer_cast(particles.device_dprops.at("eps_xy").data());
            double* d_history = thrust::raw_pointer_cast(particles.device_dprops.at("history").data());
                
            thrust::for_each(thrust::device, first_p, last_p, 
                [d_Lxx, d_Lyy, d_Lxy, d_Lyx, d_eps_xx, d_eps_yy, d_eps_xy, d_history, dt, lambda_lame, mu_lame] __device__ (idx_t p) {
                    d_eps_xx[p] += d_Lxx[p] * dt;
                    d_eps_yy[p] += d_Lyy[p] * dt;
                    d_eps_xy[p] += 0.5 * (d_Lxy[p] + d_Lyx[p]) * dt;

                    const double e_xx = d_eps_xx[p];
                    const double e_yy = d_eps_yy[p];
                    const double e_xy = d_eps_xy[p];

                    // Calcolo autovalori per decomposizione 
                    const double tr = e_xx + e_yy;   
                    const double radice = sqrt(fmax(0.0, e_xx*e_xx + e_yy*e_yy - 2.0*e_xx*e_yy + 4.0*e_xy*e_xy));    
                    const double lambda1 = fmax((tr + radice) / 2.0, 0.0);
                    const double lambda2 = fmax((tr - radice) / 2.0, 0.0);
                    const double trp = fmax(tr, 0.0);   
                    const double psi_plus = 0.5 * lambda_lame * trp * trp + mu_lame * (lambda1 * lambda1 + lambda2 * lambda2);

                    // Condizioni di Kuhn-Tucker
                    if (psi_plus > d_history[p]) {
                        d_history[p] = psi_plus;
                    }
                }
            );
            
            //Fase 4: Phase field con metodo di Jacobi
            const int j_iter=3;
            const double omega=0.8;

            // Ciclo di Jacobi
            for(int i=0; i<j_iter; i++){
                thrust::fill(particles.device_grid_vars.at("R_c").begin(), particles.device_grid_vars.at("R_c").end(), 0.0);
                thrust::fill(particles.device_grid_vars.at("D_c").begin(), particles.device_grid_vars.at("D_c").end(), 0.0);
                thrust::fill(particles.device_dprops.at("cp").begin(), particles.device_dprops.at("cp").end(), 0.0);
                thrust::fill(particles.device_dprops.at("gcx").begin(), particles.device_dprops.at("gcx").end(), 0.0);
                thrust::fill(particles.device_dprops.at("gcy").begin(), particles.device_dprops.at("gcy").end(), 0.0);
                
                particles.g2p (grid_vars, {"g_c"}, {"cp"}, false);
                particles.g2pd(grid_vars, {"g_c"}, {"gcx"}, {"gcy"}, false);
               
                const double* d_px = thrust::raw_pointer_cast(particles.device_x.data());
                const double* d_py = thrust::raw_pointer_cast(particles.device_y.data());
                const double* d_area = thrust::raw_pointer_cast(particles.device_dprops.at("area").data());
                const double* d_history = thrust::raw_pointer_cast(particles.device_dprops.at("history").data());
                const double* d_cp = thrust::raw_pointer_cast(particles.device_dprops.at("cp").data());                    const double* d_gcx = thrust::raw_pointer_cast(particles.device_dprops.at("gcx").data());
                const double* d_gcy = thrust::raw_pointer_cast(particles.device_dprops.at("gcy").data());
                const idx_t* d_ptcl_to_grd = thrust::raw_pointer_cast(particles.device_ptcl_to_grd.data());

                double* d_Rc = thrust::raw_pointer_cast(particles.device_grid_vars.at("R_c").data());
                double* d_Dc = thrust::raw_pointer_cast(particles.device_grid_vars.at("D_c").data());
                    
                const auto grid_num_rows = grid.num_rows(); 

                //ciclo particelle con atomicAdd e calcolo forme integrato
                thrust::for_each(thrust::device, first_p, last_p, 
                    [d_px, d_py, d_area, d_history, d_cp, d_gcx, d_gcy, d_ptcl_to_grd, d_Rc, d_Dc,
                    k, l0, Gc, hx, hy, grid_num_rows] __device__ (idx_t p) {
                        const double pos_x = d_px[p], pos_y = d_py[p];
                        const double w_p = d_area[p];
                        const double H = d_history[p];
                        const double cp = d_cp[p];
                        const double gcx = d_gcx[p];                            
                        const double gcy = d_gcy[p];

                        const idx_t cell_idx = d_ptcl_to_grd[p];
                        const int c = cell_idx / grid_num_rows; 
                        const int r = cell_idx % grid_num_rows; 
                            
                        const double Fp = (4.0 * l0 * (1.0 - k) * H / Gc) + 1.0;

                        //coordinate locali particella
                        const double xi = (pos_x - c * hx) / hx;
                        const double eta = (pos_y - r * hy) / hy;

                        //indici globali 4 nodi della cella
                        const idx_t nodi[4] = {
                            c * (grid_num_rows + 1) + r,
                            c * (grid_num_rows + 1) + (r + 1),
                            (c + 1) * (grid_num_rows + 1) + r,
                            (c + 1) * (grid_num_rows + 1) + (r + 1)
                        };

                        //funzioni di forma
                        const double N[4] = {
                            (1.0 - xi) * (1.0 - eta),
                            (1.0 - xi) * eta,
                            xi * (1.0 - eta),                                
                            xi * eta
                            };
                        
                        //derivate spaziali
                        const double dNx_arr[4] = {
                            (-1.0 / hx) * (1.0 - eta),
                            (-1.0 / hx) * eta,
                            ( 1.0 / hx) * (1.0 - eta),
                            ( 1.0 / hx) * eta
                        };
                        const double dNy_arr[4] = {                                
                            (1.0 - xi) * (-1.0 / hy),
                            (1.0 - xi) * ( 1.0/ hy),
                            xi * (-1.0 / hy),
                            xi * ( 1.0 / hy)
                        };

                        for (int j = 0; j < 4; ++j) {
                            const double F_nodi = N[j] * w_p;
                            const double K_nodi = (Fp * N[j] * cp + 4.0 * l0 * l0 * (gcx * dNx_arr[j] + gcy * dNy_arr[j])) * w_p;
                            atomicAdd(&d_Rc[nodi[j]], F_nodi - K_nodi);
                            atomicAdd(&d_Dc[nodi[j]], (Fp * N[j] * N[j] + 4.0 * l0 * l0 * (dNx_arr[j] * dNx_arr[j] + dNy_arr[j] * dNy_arr[j])) * w_p);
                        }
                    }
                );

                double* d_gc = thrust::raw_pointer_cast(particles.device_grid_vars.at("g_c").data());
                const double* d_gm_jacobi = thrust::raw_pointer_cast(particles.device_grid_vars.at("gm").data());

                //ciclo nodi: aggiornamento nodale di jacobi 
                thrust::for_each(thrust::device, first_n, last_n, 
                    [d_gc, d_Rc, d_Dc, d_gm_jacobi, m_tol, omega] __device__ (idx_t n) {
                        if (d_gm_jacobi[n] > m_tol && d_Dc[n] > 1e-12) {
                            double c_new = d_gc[n] + omega * (d_Rc[n] / d_Dc[n]);
                            if (isfinite(c_new)) {
                                c_new = fmax(0.0, fmin(c_new, d_gc[n]));
                                d_gc[n] = c_new;
                            } 
                        }
                    }
                );
            } // fine ciclo Jacobi
            
            // Pulizia e proiezioni finali post-Jacobi
            thrust::fill(particles.device_dprops.at("cp").begin(), particles.device_dprops.at("cp").end(), 0.0);
            thrust::fill(particles.device_dprops.at("gcx").begin(), particles.device_dprops.at("gcx").end(), 0.0);
            thrust::fill(particles.device_dprops.at("gcy").begin(), particles.device_dprops.at("gcy").end(), 0.0);

            particles.g2p (grid_vars, {"g_c"}, {"cp"}, false);
            particles.g2pd(grid_vars, {"g_c"}, {"gcx"}, {"gcy"}, false);

            //Fase 5: Calcolo degli sforzi e della funzione di degradazione
            double E_ela = 0.0;
            double E_frac = 0.0;
            double punta_y = fine_frattura_y;
            long n_rotte = 0;
            
            static thrust::device_vector<double> d_E_ela_vec(particles.num_particles);
            static thrust::device_vector<double> d_E_frac_vec(particles.num_particles);
            static thrust::device_vector<double> d_punta_y_vec(particles.num_particles);
            static thrust::device_vector<long> d_rotte_vec(particles.num_particles);

            const double* d_cp  = thrust::raw_pointer_cast(particles.device_dprops.at("cp").data());
            const double* d_gcx = thrust::raw_pointer_cast(particles.device_dprops.at("gcx").data());
            const double* d_gcy = thrust::raw_pointer_cast(particles.device_dprops.at("gcy").data());
            const double* d_area = thrust::raw_pointer_cast(particles.device_dprops.at("area").data());
            const double* d_eps_xx_5 = thrust::raw_pointer_cast(particles.device_dprops.at("eps_xx").data());
            const double* d_eps_yy_5 = thrust::raw_pointer_cast(particles.device_dprops.at("eps_yy").data());
            const double* d_eps_xy_5 = thrust::raw_pointer_cast(particles.device_dprops.at("eps_xy").data());
            const double* d_py_pos = thrust::raw_pointer_cast(particles.device_y.data());

            double* d_damage = thrust::raw_pointer_cast(particles.device_dprops.at("damage").data());
            double* d_sig_xx = thrust::raw_pointer_cast(particles.device_dprops.at("sig_xx").data());
            double* d_sig_yy = thrust::raw_pointer_cast(particles.device_dprops.at("sig_yy").data());
            double* d_sig_xy = thrust::raw_pointer_cast(particles.device_dprops.at("sig_xy").data());
            double* d_minus_sxx = thrust::raw_pointer_cast(particles.device_dprops.at("minus_sxx").data());
            double* d_minus_syy = thrust::raw_pointer_cast(particles.device_dprops.at("minus_syy").data());
            double* d_minus_sxy = thrust::raw_pointer_cast(particles.device_dprops.at("minus_sxy").data());

            double* r_E_ela  = thrust::raw_pointer_cast(d_E_ela_vec.data());
            double* r_E_frac = thrust::raw_pointer_cast(d_E_frac_vec.data());
            double* r_punta_y= thrust::raw_pointer_cast(d_punta_y_vec.data());
            long* r_rotte    = thrust::raw_pointer_cast(d_rotte_vec.data());

            thrust::for_each(thrust::device, first_p, last_p,
                [d_cp, d_gcx, d_gcy, d_area, d_eps_xx_5, d_eps_yy_5, d_eps_xy_5, d_py_pos,
                d_damage, d_sig_xx, d_sig_yy, d_sig_xy, d_minus_sxx, d_minus_syy, d_minus_sxy,
                r_E_ela, r_E_frac, r_punta_y, r_rotte, Gc, l0, k, lambda_lame, mu_lame, fine_frattura_y] __device__ (idx_t p) {

                    double cp = fmax(0.0, fmin(d_cp[p], 1.0)); 
                    double gcx = d_gcx[p], gcy = d_gcy[p];
                    const double w_p = d_area[p];
                    const double d = 1.0 - cp;
                    d_damage[p] = d;
                    long is_broken = 0;
                    double local_punta_y = fine_frattura_y;
                    if (d > 0.9) {
                        is_broken = 1;
                        local_punta_y = fmin(fine_frattura_y, d_py_pos[p]); // Punta y scende
                    }
                    r_rotte[p] = is_broken;
                    r_punta_y[p] = local_punta_y;

                    r_E_frac[p] = Gc * ( ((1.0 - cp) * (1.0 - cp)) / (4.0 * l0) + l0 * (gcx * gcx + gcy * gcy) ) * w_p;
                    const double g_deg = (1.0 - k) * cp * cp + k;
                    const double exx = d_eps_xx_5[p], eyy = d_eps_yy_5[p], exy = d_eps_xy_5[p];
                        
                    const double tr = exx + eyy;
                    const double radice = sqrt(fmax(0.0, exx*exx + eyy*eyy - 2.0*exx*eyy + 4.0*exy*exy));
                        
                    const double lambda1 = (tr + radice) / 2.0;
                    const double lambda2 = (tr - radice) / 2.0;
                        
                    const double trp = fmax(tr, 0.0);
                    const double trm = fmin(tr, 0.0);
                    const double lambda1p = fmax(lambda1, 0.0);
                    const double lambda1m = fmin(lambda1, 0.0);
                    const double lambda2p = fmax(lambda2, 0.0);
                    const double lambda2m = fmin(lambda2, 0.0);

                    const double psi_plus = 0.5 * lambda_lame * trp * trp + mu_lame * (lambda1p * lambda1p + lambda2p * lambda2p);
                    const double psi_minus= 0.5 * lambda_lame * trm * trm + mu_lame * (lambda1m * lambda1m + lambda2m * lambda2m);
                        
                    double P1xx = 1.0, P1yy = 0.0, P1xy = 0.0;
                    if (radice > 1e-16) {
                        const double cos2 = (exx - eyy) / radice;
                        const double sin2 = (2.0 * exy) / radice;
                        P1xx = 0.5 * (1.0 + cos2); 
                        P1yy = 0.5 * (1.0 - cos2); 
                        P1xy = 0.5 * sin2;
                    }                       
                    const double eposxx = lambda1p * P1xx + lambda2p * (1.0 - P1xx);
                    const double eposyy = lambda1p * P1yy + lambda2p * (1.0 - P1yy);
                    const double eposxy = (lambda1p - lambda2p) * P1xy;
                    const double eminxx = exx - eposxx;
                    const double eminyy = eyy - eposyy;
                    const double eminxy = exy - eposxy;

                    double sxx = g_deg * (lambda_lame * trp + 2.0 * mu_lame * eposxx) + (lambda_lame * trm + 2.0 * mu_lame * eminxx);
                    double syy = g_deg * (lambda_lame * trp + 2.0 * mu_lame * eposyy) + (lambda_lame * trm + 2.0 * mu_lame * eminyy);
                    double sxy = g_deg * (2.0 * mu_lame * eposxy) + (2.0 * mu_lame * eminxy);

                    r_E_ela[p] = (g_deg * psi_plus + psi_minus) * w_p;
                       
                    d_sig_xx[p] = sxx; 
                    d_sig_yy[p] = syy; 
                    d_sig_xy[p] = sxy;
                       
                    d_minus_sxx[p] = -sxx; 
                    d_minus_syy[p] = -syy;
                    d_minus_sxy[p] = -sxy;
                }
            );

            // Riduzioni finali 
            E_ela   = thrust::reduce(d_E_ela_vec.begin(), d_E_ela_vec.end(), 0.0, thrust::plus<double>());
            E_frac  = thrust::reduce(d_E_frac_vec.begin(), d_E_frac_vec.end(), 0.0, thrust::plus<double>());
            n_rotte = thrust::reduce(d_rotte_vec.begin(), d_rotte_vec.end(), 0L, thrust::plus<long>());
            // Troviamo il minimo y 
            punta_y = thrust::reduce(d_punta_y_vec.begin(), d_punta_y_vec.end(), fine_frattura_y, thrust::minimum<double>());
            
            //trasferiamo tutto alla griglia per calcolare le forze interne 
            particles.p2gd(grid_vars, {"minus_sxx", "minus_sxy"}, {"minus_sxy", "minus_syy"}, "area", {"gfx", "gfy"}, false);

            //Fase 6: F=m*a
            //aggiunta forza esterna
            static thrust::device_vector<double> d_W_ext_vec_6(nn);
            
            double* d_gfx_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gfx").data());
            double* d_gfy_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gfy").data());
            double* d_gpx_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gpx").data());
            double* d_gpy_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gpy").data());
            double* d_gvx_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gvx").data());
            double* d_gvy_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gvy").data());
            double* d_gax_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gax").data());
            double* d_gay_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gay").data());
            const double* d_gm_6 = thrust::raw_pointer_cast(particles.device_grid_vars.at("gm").data());
            
            const double* d_Fext_ptr = thrust::raw_pointer_cast(dev_Fext.data());
            double* r_W_ext = thrust::raw_pointer_cast(d_W_ext_vec_6.data());

            thrust::for_each(thrust::device, first_n, last_n,
                [d_gfx_6, d_gfy_6, d_gpx_6, d_gpy_6, d_gvx_6, d_gvy_6, d_gax_6, d_gay_6, d_gm_6, d_Fext_ptr, r_W_ext, dt, m_tol] __device__ (idx_t n) {
                    // Aggiunta forza esterna al nodo
                    d_gfx_6[n] += d_Fext_ptr[n];

                    if (d_gm_6[n] > m_tol) {
                        d_gpx_6[n] += dt * d_gfx_6[n];
                        d_gpy_6[n] += dt * d_gfy_6[n];
                        
                        double vx_new = d_gpx_6[n] / d_gm_6[n];
                        d_gvx_6[n] = vx_new;
                        d_gvy_6[n] = d_gpy_6[n] / d_gm_6[n];
                        
                        d_gax_6[n] = d_gfx_6[n] / d_gm_6[n];
                        d_gay_6[n] = d_gfy_6[n] / d_gm_6[n];
                        
                        // Lavoro infinitesimo del nodo
                        r_W_ext[n] = d_Fext_ptr[n] * vx_new * dt;
                    } else {
                        d_gfx_6[n] = d_gfy_6[n] = 0.0;
                        d_gvx_6[n] = d_gvy_6[n] = 0.0;
                        d_gax_6[n] = d_gay_6[n] = 0.0;
                        r_W_ext[n] = 0.0;
                    }
                }
            );

            // Riduciamo e accumuliamo il lavoro esterno di questa iterazione
            double W_ext_step = thrust::reduce(d_W_ext_vec_6.begin(), d_W_ext_vec_6.end(), 0.0, thrust::plus<double>());
            W_ext += W_ext_step;

            //Fase 7: Calcolo nuove velocità (PIC/FLIP) e posizioni
            thrust::fill(particles.device_dprops.at("vpx").begin(), particles.device_dprops.at("vpx").end(), 0.0);
            thrust::fill(particles.device_dprops.at("vpy").begin(), particles.device_dprops.at("vpy").end(), 0.0);
            thrust::fill(particles.device_dprops.at("ax").begin(), particles.device_dprops.at("ax").end(), 0.0);
            thrust::fill(particles.device_dprops.at("ay").begin(), particles.device_dprops.at("ay").end(), 0.0);

            particles.g2p(grid_vars, {"gvx", "gvy", "gax", "gay"}, {"vpx", "vpy", "ax", "ay"}, false);

            double E_cin = 0.0;
       
            static thrust::device_vector<double> d_E_cin_vec_7(particles.num_particles);
            double* r_E_cin_7 = thrust::raw_pointer_cast(d_E_cin_vec_7.data());
            
            double* d_vx_7 = thrust::raw_pointer_cast(particles.device_dprops.at("vx").data());
            double* d_vy_7 = thrust::raw_pointer_cast(particles.device_dprops.at("vy").data());
            double* d_px_7 = thrust::raw_pointer_cast(particles.device_x.data());
            double* d_py_7 = thrust::raw_pointer_cast(particles.device_y.data());
            
            const double* d_vpx_7 = thrust::raw_pointer_cast(particles.device_dprops.at("vpx").data());
            const double* d_vpy_7 = thrust::raw_pointer_cast(particles.device_dprops.at("vpy").data());
            const double* d_ax_7 = thrust::raw_pointer_cast(particles.device_dprops.at("ax").data());
            const double* d_ay_7 = thrust::raw_pointer_cast(particles.device_dprops.at("ay").data());
            const double* d_massa_7 = thrust::raw_pointer_cast(particles.device_dprops.at("massa").data());

            //aggiornamento velocità particelle pic/flip
            thrust::for_each(thrust::device, first_p, last_p,
                [r_E_cin_7, d_vx_7, d_vy_7, d_vpx_7, d_vpy_7, d_ax_7, d_ay_7, d_px_7, d_py_7, d_massa_7, alpha_flip, dt] __device__ (idx_t p) {
                    
                    const double vfx = d_vx_7[p] + dt * d_ax_7[p];
                    const double vfy = d_vy_7[p] + dt * d_ay_7[p];

                    const double vpx = d_vpx_7[p];
                    const double vpy = d_vpy_7[p];

                    const double new_vx = alpha_flip * vfx + (1.0 - alpha_flip) * vpx;
                    const double new_vy = alpha_flip * vfy + (1.0 - alpha_flip) * vpy;

                    d_vx_7[p] = new_vx;
                    d_vy_7[p] = new_vy;

                    d_px_7[p] += dt * vpx;
                    d_py_7[p] += dt * vpy;

                    r_E_cin_7[p] = 0.5 * d_massa_7[p] * (new_vx * new_vx + new_vy * new_vy);
                }
            );

            E_cin = thrust::reduce(d_E_cin_vec_7.begin(), d_E_cin_vec_7.end(), 0.0, thrust::plus<double>());

        
            // Fase 8: Aggiornamento variabili di controllo    
            const double t_s = step * dt;
            const double t_us = t_s * 1.0e6;
            const double E_bal = E_cin + E_ela + E_frac - W_ext;
            
            double frac_ratio = 0.0;
            double v_inst = 0.0;
            
            if (step > 0) {
                // Calcolo la velocità confrontando la posizione vecchia con la nuova (punta_y)
                v_inst = (fine_frattura_y_prev - punta_y) / dt * 1.0e-3; 
            }

            double alpha = 0.01; 
            v_tip_smoothed = alpha * v_inst + (1.0 - alpha) * v_tip_smoothed;
            
            // Aggiornamento variabili loop
            fine_frattura_y_prev = punta_y;
            t_prec = t_s;
            const double a_crack = inizio_frattura_y - punta_y;

            if (a_crack > 1e-9) frac_ratio = E_frac / (Gc * a_crack);
            
            // SALVATAGGIO CSV    
            en_file << step << ',' << t_us << ',' << E_cin << ',' << E_ela << ','
                    << E_frac << ',' << W_ext << ',' << E_bal << ',' << a_crack
                    << ',' << v_tip_smoothed << ',' << n_rotte << ','
                    << frac_ratio << '\n';

            if (step % 20 == 0) en_file.flush();  
            if (step == 1 || step % 50 == 0) {
                auto t_now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> elapsed_step = t_now - t_start_print;

                std::cout << std::left
                        << std::setw(8)  << step
                        << std::setw(12) << t_us
                        << std::setw(15) << E_bal
                        << std::setw(15) << E_frac
                        << std::setw(12) << a_crack
                        << std::setw(12) << v_tip_smoothed
                        << std::setw(10) << elapsed_step.count() << "\n";
                    
                t_start_print = t_now; 
            }
   
            const int dump_every = std::max(1, static_cast<int>(std::lround(2.5e-6 / dt)));
            if (step == 1 || step % dump_every == 0) {
        
                particles.memcpy_device_to_host();
            
                
                std::ostringstream fn;
                fn << out_dir << "/PF_DYN_" << std::setfill('0') << std::setw(6) << step << ".csv";
                std::ofstream out(fn.str());
                if (out.is_open()) {
                    particles.print<particles_t::output_format::csv>(out);
                    out.close();
                }
            }
        } // CHIUSURA DEL CICLO TEMPORALE 
    
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "Errore CUDA rilevato: " << cudaGetErrorString(err) << "\n";
        }
        std::cout << "\nSimulazione completata con successo! Premi INVIO per uscire.";
        std::cin.get();
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "\n\n[ERRORE CRITICO C++]: " << e.what() << "\n";
        std::cout << "Premi INVIO per chiudere il terminale...";
        std::cin.get();
        return 1;
    }
    catch (...) {
        std::cerr << "\n\n[ERRORE CRITICO SCONOSCIUTO (possibile crash GPU/CUDA)]\n";
        std::cout << "Premi INVIO per chiudere il terminale...";
        std::cin.get();
        return 1;
    }
}