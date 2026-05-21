module Logistic_regression
    implicit none

    private 
    public fit , standardize , sigmoid, accuracy
    
contains
subroutine fit(y,x,size_of_data, rate_of_learn, n_feature , weight , bias,iteration ,loss,mse,lamda)
    use, intrinsic :: iso_fortran_env

    implicit none
    real(real64), intent(inout) :: y(:) ,x(: , :), bias 
    real(real64), intent(inout) :: loss , mse
    real(real64), intent(in) :: rate_of_learn , lamda
    integer(int64), intent(in) :: size_of_data, iteration, n_feature
    integer(int64) :: i
    real(real64), intent(inout) :: weight(:)
    real(real64) :: dw(n_feature), db , last_loss
    real(real64), allocatable :: y_predicted(:) 

    last_loss = 0.0_real64
    weight(:) = 0.0_real64
    bias = 0.0_real64
    
    if (allocated(y_predicted)) deallocate(y_predicted)
    allocate(y_predicted(size_of_data))

    do i = 1,iteration
        y_predicted = sigmoid(matmul(x,weight) + bias)

        dw = (1.0_real64/size_of_data)*matmul(transpose(x), y_predicted - y) + &
             (lamda/size_of_data)*weight
             
        db = (1.0_real64/size_of_data)*sum(y_predicted - y)

        weight = weight - rate_of_learn*dw
        bias = bias - rate_of_learn*db
        loss = -sum(y * log(y_predicted + 1.0e-15_real64) + (1.0_real64 - y) * log(1.0_real64 - y_predicted + 1.0e-15_real64)) / size_of_data
        if ( mod(i,500) .eq. 0 ) then
            print *, loss
        end if

        if ( abs(last_loss - loss) .lt. 10e-8 ) then
            exit
        end if

        last_loss = loss
    end do

    y_predicted = sigmoid(matmul(x,weight) + bias)
    mse = sum((y_predicted - y)**2)/size_of_data

    loss = -sum(y * log(y_predicted + 1.0e-15_real64) + (1.0_real64 - &
           y) * log(1.0_real64 - y_predicted + & 
           1.0e-15_real64)) / size_of_data + &
           (lamda/size_of_data)*sum(weight**2)

    deallocate(y_predicted)

end subroutine fit

subroutine standardize(x,n_feature, size_of_data,mean,sd , z_score)
    use, intrinsic :: iso_fortran_env
    implicit none
    real(real64), intent(in) :: x(: , :)
    real(real64),  allocatable :: z_score(: , :)
    integer(int64), intent(in) :: n_feature, size_of_data
    real(real64), intent(inout) :: mean(n_feature) , sd(n_feature)
    integer(int64) :: j

    if (allocated(z_score)) deallocate(z_score)
    allocate(z_score(size_of_data,n_feature))

    do concurrent(j = 1:n_feature)
    mean(j) = (1.0_real64/size_of_data)*sum(x(:,j))
    sd(j) = sqrt(sum((x(:,j) - mean(j))**2)/size_of_data)

    if (sd(j) .gt. 1.0e-12_real64 ) then
        z_score(:,j) = (x(:,j) - mean(j))/sd(j)
    else 
        z_score(:,j) = 0.0_real64      
    end if
    end do

end subroutine standardize

elemental function sigmoid(z) result(sigma)
    use, intrinsic :: iso_fortran_env

    implicit none
    real(real64), intent(in) :: z
    real(real64):: sigma

    sigma = 1.0_real64/(1.0_real64 + exp(-z))
end function 

function accuracy(x_scaled, y,weight,bias, size_of_sample) result(acc_data)
    use, intrinsic :: iso_fortran_env

    implicit none
    integer(int64), intent(in) :: size_of_sample
    real(real64), intent(in) :: x_scaled (:,:), y(:) , weight (:) , bias
    real(real64) :: acc_data(3), y_predicted(size_of_sample)
    integer(int64) :: i, tp,tn,fp,fn
    
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    y_predicted = sigmoid(matmul(x_scaled,weight) + bias)

    do  i = 1,size_of_sample
        if ( y_predicted(i) .ge. 0.5 .and. y(i) .eq. 1 ) then
            tp = tp + 1
        else if (y_predicted(i) .lt. 0.5 .and. y(i) .eq. 0 ) then
            tn = tn + 1
        else if (y_predicted(i) .ge. 0.5 .and. y(i) .eq. 0) then
            fp = fp + 1
        else if (y_predicted(i) .lt. 0.5 .and. y(i) .eq. 1) then
            fn = fn + 1
        end if
    end do

    acc_data(1) = real(tp + tn)/real(size_of_sample)
    if ( tp + fp .eq. 0 ) then
        acc_data(2) = 0.0
    else
        acc_data(2) = real(tp) / real(tp + fp)
    end if
    if (tp + fn .ne. 0) then
        acc_data(3) = real(tp) / real(tp + fn)
    else 
        acc_data(3) = 0.0
    end if
end function accuracy

end module Logistic_regression
