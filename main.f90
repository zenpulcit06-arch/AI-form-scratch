program main
    use, intrinsic :: iso_fortran_env
    use network

    implicit none
    integer(int64) :: n_feature,sample_size, iteration,i,j,mode,N, label
    integer(int64), allocatable :: neurons(:)
    real(real64) :: learning_rate,loss,acc,start,finish
    real(real64), allocatable :: x(:,:), y(:,:) 
    character(len = 1000) :: filename
    type(layer), allocatable :: net(:)

    print *, "Enter no. of feature"
    read(*,*) n_feature

    print *, "Enter sample size"
    read(*,*) sample_size

    print *, 'Enter no. of iteration'
    read(*,*) iteration

    print *, 'Enter learning rate'
    read(*,*) learning_rate

    print *, 'Enter no. of layers'
    read(*,*) N

    if (allocated(net)) deallocate(net)
    allocate(net(N))
    
    if (allocated(neurons)) deallocate(neurons)
    allocate(neurons(N))

    do i = 1, N
        print *, 'Enter no. of neurons in layer',i
        read(*,*) neurons(i)
    end do

    call intialize_layer(net(1),n_feature,neurons(1),sample_size)

    do i = 2,N
        call intialize_layer(net(i),neurons(i-1),neurons(i),sample_size)
    end do

    if (allocated(x)) deallocate(x)
    allocate(x(sample_size, n_feature))
    if(allocated(y)) deallocate(y)
    allocate(y(sample_size, size(net(N)%W,2)))

    mode = 0
    print *, 'Enter 1 for manual entry or 2 for CSV entry:'
    read(*,*) mode

    if (mode == 2) then
        print *, 'Enter Filename:'
        read(*,*) filename
        print *, 'Reading data from ', trim(filename), '...'
        open(unit=10, file=trim(filename), status='old', action='read')
        do i = 1, sample_size
            read(10, *) x(i, :), label
            y(i,:) = 0.0_real64
            y(i,label + 1) = 1.0_real64
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
            read(*,*) y(i,1)
        end do
    end if

    call cpu_time(start)
    call fit_network(net,x,y,learning_rate,iteration,sample_size,N,loss)
    acc = accuracy(net(N)%H,y,sample_size)
    call cpu_time(finish)
    
    print *, 'Loss =',loss
    print *, 'Accuracy =', acc
    print *, 'Time of execution =', finish - start,'second'

end program main