module Layer
    use, intrinsic :: iso_fortran_env
    use Logistic_regression, only: sigmoid
    implicit none
    private
    type, public :: network
    real(real64),allocatable :: W1(:,:),W2(:,:), b1(:),b2(:),H(:,:)
    end type
    
    public intialize, forward_pass, backward_pass, update_weight, fit, relu
contains

elemental function relu(z) result(r)
    real(real64), intent(in) :: z
    real(real64) :: r
    r = max(0.0_real64, z)
end function relu

subroutine intialize(net,neurons1,neurons2,n_feature,sample_size)
    class(network),intent(inout) :: net
    integer(int64), intent(in) :: neurons1,neurons2, n_feature, sample_size

    call random_seed()

    if (allocated(net%W1)) deallocate(net%W1)
    allocate(net%W1(n_feature,neurons1))
    call random_number(net%W1)
    net%W1 = (-1.0_real64 + 2.0_real64*net%W1)*0.01_real64

    if (allocated(net%W2)) deallocate(net%W2)
    allocate(net%W2(neurons1,neurons2))
    call random_number(net%W2)
    net%W2 = (-1.0_real64 + 2.0_real64*net%W2)*0.01_real64

    if (allocated(net%b1)) deallocate(net%b1)
    allocate(net%b1(neurons1))
    net%b1(:) = 0.0_real64

    if (allocated(net%b2)) deallocate(net%b2)
    allocate(net%b2(neurons2))
    net%b2(:) = 0.0_real64

    if (allocated(net%H)) deallocate(net%H)
    allocate(net%H(sample_size,neurons1))
    net%H(:,:) = 0.0_real64

end subroutine intialize

subroutine forward_pass(net,X,Y)
    class(network), intent(inout) :: net
    real(real64), intent(inout) :: X(:,:)
    real(real64), intent(out) :: Y(:,:)

    net%H = relu(matmul(X, net%W1) + spread(net%b1, 1, size(X, 1)))
    Y = sigmoid(matmul(net%H,net%W2) + spread(net%b2,1,size(net%H,1)))
end subroutine forward_pass

subroutine backward_pass(net, dW2, sample_size,y_cap,y,dw1,X,db2,db1)
    class(network), intent(inout) :: net
    real(real64), intent(out) :: dW2(:,:), dW1(:,:), db2(:),db1(:)
    real(real64), intent(in) :: y_cap(:,:), y(:,:),X(:,:)
    integer(int64), intent(in) :: sample_size

    dW2 = matmul(transpose(net%H),(y_cap - y))/sample_size

   dW1 = matmul(transpose(X),(matmul(y_cap-y,transpose(net%W2)) * &
         merge(1.0_real64, 0.0_real64, net%H > 0.0_real64)))/ sample_size
    
    db2 = sum(y_cap - y, dim =1)/sample_size

    db1 = sum(matmul(y_cap - y, transpose(net%W2)) * merge(1.0_real64, 0.0_real64, net%H > 0.0_real64),dim=1)/sample_size

end subroutine backward_pass

subroutine update_weight(net,dW1,dW2,db1,db2,rl)
    class(network), intent(inout) :: net
    real(real64), intent(in) :: dW1(:,:), dW2(:,:), db1(:),db2(:),rl

    net%W1 = net%W1 - rl * dW1
    net%W2 = net%W2 - rl * dW2

    net%b1 = net%b1 - rl * db1
    net%b2 = net%b2 - rl * db2
end subroutine update_weight

subroutine fit(net,X,y,neurons1,neurons2,sample_size,rl,iteration,loss,n_feature)
    class(network), intent(inout):: net
    integer(int64), intent(in) :: neurons1, neurons2,sample_size,iteration,n_feature
    real(real64), intent(inout) :: X(:,:),y(:,:),loss
    real(real64), intent(in) :: rl
    real(real64),allocatable:: y_predicted(:,:),dW2(:,:),dW1(:,:),db1(:),db2(:)
    real(real64) :: last_loss
    integer(int64) :: i

    last_loss = 0.0_real64
    if (allocated(y_predicted)) deallocate(y_predicted)
    allocate(y_predicted(sample_size,neurons2))

    if (allocated(dW2)) deallocate(dW2)
    allocate(dW2(neurons1,neurons2))

    if (allocated(dW1)) deallocate(dW1)
    allocate(dW1(n_feature,neurons1))

    if (allocated(db1)) deallocate(db1)
    allocate(db1(neurons1))

    if (allocated(db2)) deallocate(db2)
    allocate(db2(neurons2))
    
    call intialize(net,neurons1,neurons2,n_feature,sample_size)
    do i = 1, iteration
        call forward_pass(net,X,y_predicted)
        loss = -sum(y * log(y_predicted + 1.0e-15) + (1.0 - y) * log(1.0 - y_predicted + 1.0e-15)) / sample_size
        if ( mod(i,100) .eq. 0 ) then
            print *, loss
        end if
        call backward_pass(net,dW2,sample_size,y_predicted,y,dW1,X,db2,db1)
        call update_weight(net,dW1,dW2,db1,db2,rl)
        if ( abs(last_loss - loss) .lt. 10e-8 ) then
            exit
        end if
        last_loss = loss
    end do
end subroutine fit
end module Layer
