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
        if ( abs(last_loss - loss) .lt. 1e-8_real64 ) then
            exit
        end if
        last_loss = loss
    end do
end subroutine fit
end module Layer


module Network
    use, intrinsic :: iso_fortran_env
    use Logistic_regression, only: sigmoid
    use Layer, only: relu
    implicit none
    private

    type :: delta_t
    real(real64), allocatable :: d(:,:)
    end type

    type, public :: layer
        real(real64), allocatable :: W(:,:), b(:), H(:,:)
    end type

    public intialize_layer, forward_pass_network, allocate_delta, backward_pass_network,fit_network, softmax, accuracy,resize_H
contains

subroutine resize_H(lay,sample_size)
    class(layer) :: lay
    integer(int64) :: sample_size

    if(allocated(lay%H)) deallocate(lay%H)
    allocate(lay%H(sample_size, size(lay%W,2)))
    lay%H(:,:) = 0.0_real64

end subroutine resize_H

pure function softmax(z) result(r)
    real(real64), intent(in)  :: z(:)
    real(real64), allocatable :: r(:)
    
    real(real64) :: c
    real(real64) :: temp_sum
    
    c = maxval(z)
    r = exp(z - c)
    
    temp_sum = sum(r)
    r = r / temp_sum
end function softmax

subroutine intialize_layer(lay, prev_neurons, current_neurons,sample_Size)
    class(layer), intent(inout) :: lay
    integer(int64), intent(in) :: prev_neurons, current_neurons,sample_Size

    call random_seed()

    if (allocated(lay%W)) deallocate(lay%W)
    allocate(lay%W(prev_neurons,current_neurons))
    call random_number(lay%W)
    lay%W = (-1.0_real64 + 2.0_real64*lay%W) * &
             sqrt(2.0_real64/real(prev_neurons,real64))

    if (allocated(lay%b)) deallocate(lay%b)
    allocate(lay%b(current_neurons))
    lay%b(:) = 0.0_real64

    if(allocated(lay%H)) deallocate(lay%H)
    allocate(lay%H(sample_Size, current_neurons))
    lay%H(:,:) = 0.0_real64
end subroutine intialize_layer

subroutine forward_pass_network(net,X,N,sample_size)
    class(layer), intent(inout) :: net(:)
    integer(int64), intent(in) :: N, sample_size
    real(real64), intent(inout):: X(:,:)
    integer(int64) :: i,j
    real(real64) :: temp_array(sample_size,size(net(N)%W,2))
    real(real64), allocatable :: tarr(:,:)

    select case (N)
    case (1)
        ! matmul(X,net(N)%W)
        call DGEMM('N', 'N', int(sample_size), int(size(net(N)%W,2)), int(size(net(N)%W,1)), &
           1.0d0, X, int(sample_size), net(N)%W, int(size(net(N)%W,1)), &
           0.0d0, temp_array, int(sample_size))

        temp_array = temp_array + spread(net(N)%b,1,size(X,1))
        
        do j = 1,sample_size
            net(N)%H(j,:) = softmax(temp_array(j,:))
        end do

    case default

        if (allocated(tarr)) deallocate(tarr)
        allocate(tarr(sample_size, size(net(1)%W,2)))

        !matmul(X,net(1)%W)
        call DGEMM('N', 'N', int(sample_size), int(size(net(1)%W,2)), int(size(net(1)%W,1)), &
           1.0d0, X, int(sample_size), net(1)%W, int(size(net(1)%W,1)), &
           0.0d0, tarr, int(sample_size))
        net(1)%H = relu(tarr + spread(net(1)%b,1,size(X,1))) 

    do i = 2, N-1

        if (allocated(tarr)) deallocate(tarr)
        allocate(tarr(sample_size, size(net(i)%W,2)))

        !matmul(net(i-1)%H,net(i)%W)
        call DGEMM('N', 'N', int(sample_size), int(size(net(i)%W,2)), int(size(net(i)%W,1)), &
           1.0d0, net(i-1)%H, int(sample_size), net(i)%W, int(size(net(i)%W,1)), &
           0.0d0, tarr, int(sample_size))
        
        net(i)%H = relu(tarr + spread(net(i)%b,1,&
            size(net(i-1)%H,1)))
    end do

    !matmul(net(N-1)%H,net(N)%W)
    call DGEMM('N', 'N', int(sample_size), int(size(net(N)%W,2)), int(size(net(N)%W,1)), &
           1.0d0, net(N-1)%H, int(sample_size), net(N)%W, int(size(net(N)%W,1)), &
           0.0d0, temp_array, int(sample_size))

    temp_array = temp_array + spread(net(N)%b,1,&
            size(net(N-1)%H,1))

    do i = 1,sample_size
        net(N)%H(i,:) = softmax(temp_array(i,:))
    end do

    end select
end subroutine forward_pass_network

subroutine allocate_delta(net,delta,sample_size,N)
    class(delta_t), intent(inout) :: delta(:)
    class(layer), intent(in) :: net(:)
    integer(int64), intent(in) :: sample_size,N
    integer(int64):: i

    do i = 1, N
        if (allocated(delta(i)%d)) deallocate(delta(i)%d)
        allocate(delta(i)%d(sample_size,size(net(i)%W,2)))
    end do
end subroutine allocate_delta

subroutine backward_pass_network(delta,net,X,y,y_cap,N,sample_size,rl,lamda)
    class(delta_t), intent(inout) :: delta(:)
    class(layer), intent(inout) :: net(:)
    real(real64), intent(inout) :: X(:,:),y(:,:), y_cap(:,:)
    integer(int64), intent(in) :: sample_size, N
    integer(int64) :: i
    real(real64), intent(in) :: rl
    real(real64), allocatable :: dw(:,:),db(:)
    real(real64), intent(in), optional :: lamda
    real(real64) :: lam

    if (present(lamda)) then
        lam = lamda
    else
        lam = 0.0_real64
    end if
    
    delta(N)%d = y_cap - y

    do i = N -1, 1,-1 

        !matmul(delta(i+1)%d, transpose(net(i+1)%W))
        call DGEMM('N', 'T', int(sample_size), int(size(net(i+1)%W,1)), int(size(net(i+1)%W,2)), &
           1.0d0, delta(i+1)%d, int(sample_size), net(i+1)%W, int(size(net(i+1)%W,1)), &
           0.0d0, delta(i)%d, int(sample_size))

        delta(i)%d = delta(i)%d * merge(1.0_real64, 0.0_real64, net(i)%H > 0.0_real64)
    end do

    do i = 1,N
        if (allocated(dw)) deallocate(dw)
        allocate(dw(size(net(i)%W,1), size(net(i)%W,2)))

        if (allocated(db)) deallocate(db)
        allocate(db(size(delta(i)%d,2)))

        if (i .ne. 1) then
            !matmul(transpose(net(i-1)%H),delta(i)%d)
            call DGEMM('T', 'N', int(size(net(i-1)%H,2)), int(size(delta(i)%d,2)), int(sample_size), &
               1.0d0, net(i-1)%H, int(sample_size), delta(i)%d, int(sample_size), &
               0.0d0, dw, int(size(net(i-1)%H,2)))

            dw = dw/sample_size + lam*net(i)%W/sample_size
        else
            !matmul(transpose(X),delta(i)%d)
            call DGEMM('T', 'N', int(size(X,2)), int(size(delta(i)%d,2)), int(sample_size), &
               1.0d0, X, int(sample_size), delta(i)%d, int(sample_size), &
               0.0d0, dw, int(size(X,2)))
            dw = dw/sample_size + lam* net(i)%W/sample_size
        end if
        db = sum(delta(i)%d,dim =1)/sample_size 
        net(i)%W = net(i)%W - dw*rl
        net(i)%b = net(i)%b - db*rl
    end do

end subroutine backward_pass_network

subroutine fit_network(net,X,y,rl,iteration,sample_size,N,loss,lamda)
    class(layer), intent(inout) :: net(:)
    class(delta_t), allocatable :: delta(:)
    real(real64), intent(inout) :: X(:,:),y(:,:),loss
    real(real64), intent(in) :: rl
    real(real64), allocatable :: y_predicted(:,:)
    integer(int64),intent(in) :: N,iteration,sample_size
    integer(int64) :: i
    real(real64) :: last_loss, lam
    real(real64), intent(in), optional :: lamda

    if (present(lamda)) then
        lam = lamda
    else
        lam = 0.0_real64
    end if

    last_loss = 0.0_real64

    if (allocated(delta)) deallocate(delta)
    allocate(delta(N))
    call allocate_delta(net, delta, sample_size, N)

    if (allocated(y_predicted)) deallocate(y_predicted)
    allocate(y_predicted(sample_size,size(net(N)%W,2)))

    do i = 1, iteration
        call forward_pass_network(net,X,N,sample_size)

        y_predicted = net(N)%H
        
        loss = -sum(y * log(y_predicted + 1.0e-15_real64)) / sample_size
        
        call backward_pass_network(delta,net,X,y,y_predicted,N,sample_size,rl,lam)
        if ( abs(last_loss - loss) .lt. 1e-8_real64 ) then
            exit
        end if
        last_loss = loss
    end do
end subroutine fit_network

function accuracy(y_predicted,y,sample_size) result(acc)
    real(real64), intent(in) :: y_predicted(:,:) , y(:,:)
    integer(int64), intent(in) :: sample_size
    real(real64) :: acc
    integer(int64):: i,right

    right = 0
    do i = 1,sample_size
        if ( maxloc(y(i,:),1) .eq. maxloc(y_predicted(i,:),1) ) then
                right = right + 1
        end if
    end do

    acc = real(right,real64)/real(sample_size,real64)
end function accuracy
    
end module Network