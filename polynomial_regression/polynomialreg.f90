module Polynomial_reg
    use, intrinsic :: iso_fortran_env
    use Logistic_regression
    implicit none
    private li_sol
    public make_poly
    
contains
    function make_poly(x, degree, sample_size, n_feature) result(poly)
        implicit none
        real(real64), allocatable :: poly(:,:)
        integer(int64), allocatable :: history(:), res_matrix(:,:)
        integer(int64), intent(in) :: degree, sample_size, n_feature
        real(real64), intent(in) :: x(:,:)
        integer(int64) :: i, j, f, no_sol, total_cols, index, sol_count

        total_cols = nint(exp(log_gamma(real(degree + n_feature + 1, real64)) - &
                              log_gamma(real(n_feature + 1, real64))          - &
                              log_gamma(real(degree + 1, real64))))            - 1

        allocate(poly(sample_size, total_cols))
        allocate(history(n_feature))

        index = 1

        do i = 1, degree
            no_sol = nint(exp(log_gamma(real(i + n_feature, real64)) - &
                              log_gamma(real(n_feature, real64))      - &
                              log_gamma(real(i + 1, real64))))

            allocate(res_matrix(no_sol, n_feature))
            sol_count = 0
            history   = 0

            call li_sol(i, n_feature, res_matrix, history, sol_count)

            do j = 1, no_sol
                poly(:, index) = 1.0_real64
                do f = 1, n_feature
                    if (res_matrix(j, f) > 0) then
                        poly(:, index) = poly(:, index) * (x(:, f) ** res_matrix(j, f))
                    end if
                end do
                index = index + 1
            end do

            deallocate(res_matrix)
        end do

        deallocate(history)
    end function make_poly


    recursive subroutine li_sol(L, n, res, history, sol_count)
        integer(int64), intent(in)    :: L, n
        integer(int64), intent(inout) :: res(:,:), history(:), sol_count
        integer(int64) :: i

        if (n /= 1) then
            do i = 0, L
                history(n) = i
                call li_sol(L - i, n - 1, res, history, sol_count)
            end do
        else
            history(1)        = L
            sol_count         = sol_count + 1
            res(sol_count, :) = history
        end if
    end subroutine li_sol

end module Polynomial_reg




