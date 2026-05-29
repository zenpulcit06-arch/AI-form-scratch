module sgd
    use, intrinsic :: iso_fortran_env
    use Network
    implicit none
    
    private
    public sgd_shuffel, sgd_fit
contains
    function sgd_shuffel(x_size) result(idx)
        integer(int64),intent(in) :: x_size
        integer(int64) :: i, j, idx(x_size), temp
        real(real64) ::  r

        idx = [(i,i = 1 ,x_size)]

        do i = x_size, 2,-1
            call random_number(r)
            j = int(r*i) + 1

            temp = idx(i)
            idx(i) = idx(j)
            idx(j) = temp
        end do
    end function sgd_shuffel

    subroutine sgd_fit(net, X, y, rl, n_epochs, sample_size, N, loss, lamda, batch_size)
        class(layer), intent(inout) :: net(:)
        class(delta_t), allocatable :: delta(:)
        real(real64), intent(inout) :: X(:,:),y(:,:),loss
        real(real64), intent(in) :: rl
        integer(int64), intent(in) :: n_epochs,sample_size,N
        real(real64), intent(in), optional :: lamda
        integer(int64), intent(in), optional :: batch_size
        real(real64) :: lam, loss_accu, batch_loss
        real(real64), allocatable :: y_predicted(:,:), X_batch(:,:), y_batch(:,:)
        integer(int64) :: B,i,j,k,start,end,no_batch
        integer(int64), allocatable :: idx(:)

        if (present(lamda)) then
            lam = lamda
        else 
            lam = 0.0_real64
        end if

        if ( present(batch_size) ) then
            B = batch_size
        else 
            B = sample_size
        end if

        no_batch = int(real(sample_size)/real(B))

        if (allocated(delta)) deallocate(delta)
        allocate(delta(N))
        call allocate_delta(net, delta, B, N)

        if (allocated(y_predicted)) deallocate(y_predicted)
        allocate(y_predicted(B,size(net(N)%W,2)))

        do k = 1, N
            call resize_H(net(k), B)
        end do

        do i = 1, n_epochs
            idx = sgd_shuffel(int(size(X,1),int64))
            loss_accu = 0

            do j = 1, no_batch
                start = B*(j-1) + 1
                end = B*(j)

                X_batch = x(idx(start:end),:)
                y_batch = y(idx(start:end),:)

                call forward_pass_network(net,X_batch,N,B)

                y_predicted = net(N)%H

                batch_loss = -sum(y_batch * log(y_predicted + 1.0e-15_real64)) / B

                call backward_pass_network(delta,net,X_batch,y_batch,y_predicted,N,B,rl,lam)
                
                loss_accu = loss_accu + batch_loss
            end do
            loss = loss_accu/no_batch

            if ( mod(i,100) .eq. 0 ) then
                print *, 'Epoch = ',i,'avg loss = ',loss
            end if
        end do

        do k = 1, N
            call resize_H(net(k), sample_size)
        end do
    end subroutine sgd_fit
end module sgd
