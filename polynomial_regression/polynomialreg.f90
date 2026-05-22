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




program main
    use, intrinsic :: iso_fortran_env
    use :: Logistic_regression
    use :: Polynomial_reg
    implicit none

    real(real64), allocatable :: x(:,:), y(:), x_poly(:,:), x_scaled(:,:)
    real(real64), allocatable :: sd(:), weight(:), mean(:)
    real(real64), allocatable :: x_predict_raw(:,:), x_predict_poly(:,:), x_predict_scaled(:)

    real(real64)   :: bias, rate_of_learn, loss, mse, y_prediction, lamda
    integer(int64) :: degree, sample_size, n_feature, total_cols, iteration, mode, i, j
    character(len=1000) :: filename

    print *, 'Enter number of base features (n_feature):'
    read(*,*) n_feature

    print *, 'Enter degree of polynomial expansion:'
    read(*,*) degree

    print *, 'Enter rate of learning (alpha):'
    read(*,*) rate_of_learn

    print *, 'Enter lambda (regularization parameter):'
    read(*,*) lamda

    print *, 'Enter total training iterations:'
    read(*,*) iteration

    print *, 'Enter sample size:'
    read(*,*) sample_size

    allocate(x(sample_size, n_feature), y(sample_size))

    total_cols = nint(exp(log_gamma(real(degree + n_feature + 1, real64)) - &
                          log_gamma(real(n_feature + 1, real64))          - &
                          log_gamma(real(degree + 1, real64))))            - 1

    allocate(mean(total_cols), sd(total_cols), weight(total_cols))
    allocate(x_predict_raw(1, n_feature), x_predict_scaled(total_cols))
    allocate(x_poly(sample_size, total_cols), x_scaled(sample_size, total_cols))

    mode = 0
    print *, 'Enter 1 for manual entry or 2 for CSV entry:'
    read(*,*) mode

    if (mode == 2) then
        print *, 'Enter Filename:'
        read(*,*) filename
        print *, 'Reading data from ', trim(filename), '...'
        open(unit=10, file=trim(filename), status='old', action='read')
        do i = 1, sample_size
            read(10, *) x(i, :), y(i)
        end do
        close(10)
        print *, 'Data loaded successfully.'
    else
        do i = 1, sample_size
            print *, '--- Sample ', i, ' ---'
            do j = 1, n_feature
                print *, 'Enter feature ', j, ':'
                read(*,*) x(i, j)
            end do
            print *, 'Enter target label y (0 or 1):'
            read(*,*) y(i)
        end do
    end if

    print *, 'Expanding feature matrices via combinatorics...'
    x_poly = make_poly(x, degree, sample_size, n_feature)

    print *, 'Standardizing features across expanded mapping space...'
    call standardize(x_poly, total_cols, sample_size, mean, sd, x_scaled)

    print *, 'Fitting Weights...'
    call fit(y, x_scaled, sample_size, rate_of_learn, total_cols, weight, bias, iteration, loss, mse, lamda)

    print *, '========================================='
    print *, '        TRAINING METRICS SUMMARY         '
    print *, '========================================='
    print *, 'Final Loss : ', loss
    print *, 'Final MSE  : ', mse
    print *, 'Model Bias : ', bias
    print *, 'Weights    :', weight
    print *, '-----------------------------------------'

    do while (.true.)
        print *, 'Enter feature 1 (or -999 to exit):'
        read(*,*) x_predict_raw(1, 1)
        if (x_predict_raw(1, 1) == -999.0_real64) exit

        do j = 2, n_feature
            print *, 'Enter feature ', j, ':'
            read(*,*) x_predict_raw(1, j)
        end do

        x_predict_poly = make_poly(x_predict_raw, degree, 1_int64, n_feature)

        do concurrent (j = 1:total_cols)
            if (sd(j) > 1.0e-9_real64) then
                x_predict_scaled(j) = (x_predict_poly(1, j) - mean(j)) / sd(j)
            else
                x_predict_scaled(j) = x_predict_poly(1, j) - mean(j)
            end if
        end do

        y_prediction = sigmoid(dot_product(weight, x_predict_scaled) + bias)

        print *, 'Probability  :', y_prediction
        if (y_prediction >= 0.5_real64) then
            print *, 'Classification: POSITIVE (1)'
        else
            print *, 'Classification: NEGATIVE (0)'
        end if

        deallocate(x_predict_poly)
        print *, '-----------------------------------------'
    end do

    print *, 'Execution complete. Press Enter to exit.'
    read(*,*)

end program main